/*
  This file is part of MADNESS.

  Copyright (C) <2007> <Oak Ridge National Laboratory>

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

  For more information please contact:

  Robert J. Harrison
  Oak Ridge National Laboratory
  One Bethel Valley Road
  P.O. Box 2008, MS-6367

  email: harrisonrj@ornl.gov
  tel:   865-241-3937
  fax:   865-572-0680


  $Id$
*/

/// \file moldft.cc
/// \brief Molecular HF and DFT code

#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include <mra/mra.h>
#include <mra/lbdeux.h>
#include <misc/ran.h>
#include <linalg/solvers.h>
#include <ctime>
#include <list>
using namespace madness;

extern int x_rks_s__(const double *rho, double *f, double *dfdra);
extern int c_rks_vwn5__(const double *rho, double *f, double *dfdra);
// extern int x_uks_s__(const double *rho, double *f, double *dfdra);
// extern int c_uks_vwn5__(const double *rho, double *f, double *dfdra);

#include <moldft/molecule.h>
#include <moldft/molecularbasis.h>


/// Simple (?) version of BLAS-1 DROT(N, DX, INCX, DY, INCY, DC, DS)
void drot(long n, double* restrict a, double* restrict b, double s, double c, long inc) {
    if (inc == 1) {
        for (long i=0; i<n; i++) {
            double aa = a[i]*c - b[i]*s;
            double bb = b[i]*c + a[i]*s;
            a[i] = aa;
            b[i] = bb;
        }
    }
    else {
        for (long i=0; i<(n*inc); i+=inc) {
            double aa = a[i]*c - b[i]*s;
            double bb = b[i]*c + a[i]*s;
            a[i] = aa;
            b[i] = bb;
        }
    }
}

void drot3(long n, double* restrict a, double* restrict b, double s, double c, long inc) {
    if (inc == 1) {
        n*=3;
        for (long i=0; i<n; i+=3) {
            double aa0 = a[i  ]*c - b[i  ]*s;
            double bb0 = b[i  ]*c + a[i  ]*s;
            double aa1 = a[i+1]*c - b[i+1]*s;
            double bb1 = b[i+1]*c + a[i+1]*s;
            double aa2 = a[i+2]*c - b[i+2]*s;
            double bb2 = b[i+2]*c + a[i+2]*s;
            a[i  ] = aa0;
            b[i  ] = bb0;
            a[i+1] = aa1;
            b[i+1] = bb1;
            a[i+2] = aa2;
            b[i+2] = bb2;
        }
    }
    else {
        inc*=3;
        n*=inc;
        for (long i=0; i<n; i+=inc) {
            double aa0 = a[i  ]*c - b[i  ]*s;
            double bb0 = b[i  ]*c + a[i  ]*s;
            double aa1 = a[i+1]*c - b[i+1]*s;
            double bb1 = b[i+1]*c + a[i+1]*s;
            double aa2 = a[i+2]*c - b[i+2]*s;
            double bb2 = b[i+2]*c + a[i+2]*s;
            a[i  ] = aa0;
            b[i  ] = bb0;
            a[i+1] = aa1;
            b[i+1] = bb1;
            a[i+2] = aa2;
            b[i+2] = bb2;
        }
    }
}

class LevelPmap : public WorldDCPmapInterface< Key<3> > {
private:
    const int nproc;
public:
    LevelPmap() : nproc(0) {};

    LevelPmap(World& world) : nproc(world.nproc()) {}

    /// Find the owner of a given key
    ProcessID owner(const Key<3>& key) const {
        Level n = key.level();
        if (n == 0) return 0;
        hashT hash;
        if (n <= 3 || (n&0x1)) hash = key.hash();
        else hash = key.parent().hash();
        //hashT hash = key.hash();
        return hash%nproc;
    }
};

typedef SharedPtr< WorldDCPmapInterface< Key<3> > > pmapT;
typedef Vector<double,3> coordT;
typedef SharedPtr< FunctionFunctorInterface<double,3> > functorT;
typedef Function<double,3> functionT;
typedef vector<functionT> vecfuncT;
typedef pair<vecfuncT,vecfuncT> pairvecfuncT;
typedef vector<pairvecfuncT> subspaceT;
typedef Tensor<double> tensorT;
typedef FunctionFactory<double,3> factoryT;
typedef SeparatedConvolution<double,3> operatorT;
typedef SharedPtr<operatorT> poperatorT;

double ttt, sss;
void START_TIMER(World& world) {
		world.gop.fence(); ttt=wall_time(); sss=cpu_time();
}

void END_TIMER(World& world, const char* msg) {
	ttt=wall_time()-ttt; sss=cpu_time()-sss; if (world.rank()==0) printf("timer: %20.20s %8.2fs %8.2fs\n", msg, sss, ttt);
}


inline double mask1(double x) {
    /* Iterated first beta function to switch smoothly
       from 0->1 in [0,1].  n iterations produce 2*n-1
       zero derivatives at the end points. Order of polyn
       is 3^n.

       Currently use one iteration so that first deriv.
       is zero at interior boundary and is exactly representable
       by low order multiwavelet without refinement */

    x = (x*x*(3.-2.*x));
    return x;
}

double mask3(const coordT& ruser) {
    coordT rsim;
    user_to_sim(ruser, rsim);
    double x= rsim[0], y=rsim[1], z=rsim[2];
    double lo = 0.0625, hi = 1.0-lo, result = 1.0;
    double rlo = 1.0/lo;

    if (x<lo)
        result *= mask1(x*rlo);
    else if (x>hi)
        result *= mask1((1.0-x)*rlo);
    if (y<lo)
        result *= mask1(y*rlo);
    else if (y>hi)
        result *= mask1((1.0-y)*rlo);
    if (z<lo)
        result *= mask1(z*rlo);
    else if (z>hi)
        result *= mask1((1.0-z)*rlo);

    return result;
}

class MolecularPotentialFunctor : public FunctionFunctorInterface<double,3> {
private:
    const Molecule& molecule;
public:
    MolecularPotentialFunctor(const Molecule& molecule)
            : molecule(molecule) {}

    double operator()(const coordT& x) const {
        return molecule.nuclear_attraction_potential(x[0], x[1], x[2]);
    }
};

class MolecularGuessDensityFunctor : public FunctionFunctorInterface<double,3> {
private:
    const Molecule& molecule;
    const AtomicBasisSet& aobasis;
public:
    MolecularGuessDensityFunctor(const Molecule& molecule, const AtomicBasisSet& aobasis)
            : molecule(molecule), aobasis(aobasis) {}

    double operator()(const coordT& x) const {
        return aobasis.eval_guess_density(molecule, x[0], x[1], x[2]);
    }
};


class AtomicBasisFunctor : public FunctionFunctorInterface<double,3> {
private:
    const AtomicBasisFunction aofunc;
    vector<coordT> specialpt;
public:
    AtomicBasisFunctor(const AtomicBasisFunction& aofunc)
    : aofunc(aofunc)
    {
    	double x, y, z;
		aofunc.get_coords(x,y,z);
		coordT r;
		r[0]=x; r[1]=y; r[2]=z;
		specialpt=vector<coordT>(1,r);
    }

    double operator()(const coordT& x) const {
        return aofunc(x[0], x[1], x[2]);
    }

    vector<coordT> special_points() const {return specialpt;}
};

class MolecularDerivativeFunctor : public FunctionFunctorInterface<double,3> {
private:
    const Molecule& molecule;
    const int atom;
    const int axis;
public:
    MolecularDerivativeFunctor(const Molecule& molecule, int atom, int axis)
            : molecule(molecule), atom(atom), axis(axis) {}

    double operator()(const coordT& x) const {
        return molecule.nuclear_attraction_potential_derivative(atom, axis, x[0], x[1], x[2]);
    }
};

/// A MADNESS functor to compute either x, y, or z
class DipoleFunctor : public FunctionFunctorInterface<double,3> {
private:
    const int axis;
public:
    DipoleFunctor(int axis) : axis(axis) {}
    double operator()(const coordT& x) const {
        return x[axis];
    }
};

double rsquared(const coordT& r) {
    return r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
}

/// A MADNESS functor to compute the cartesian moment x^i * y^j * z^k (i, j, k integer and >= 0)
class MomentFunctor : public FunctionFunctorInterface<double,3> {
private:
    const int i, j, k;
public:
    MomentFunctor(int i, int j, int k) : i(i), j(j), k(k) {}
    double operator()(const coordT& r) const {
        double xi=1.0, yj=1.0, zk=1.0;
        for (int p=0; p<i; p++) xi *= r[0];
        for (int p=0; p<j; p++) yj *= r[1];
        for (int p=0; p<k; p++) zk *= r[2];
        return xi*yj*zk;
    }
};


/// Given overlap matrix, return rotation with 3rd order error to orthonormalize the vectors
tensorT Q3(const tensorT& s) {
    tensorT Q = inner(s,s);
    Q.gaxpy(0.2,s,-2.0/3.0);
    for (int i=0; i<s.dim[0]; i++) Q(i,i) += 1.0;
    return Q.scale(15.0/8.0);
}

/// Computes matrix square root (not used any more?)
tensorT sqrt(const tensorT& s, double tol=1e-8) {
    int n=s.dim[0], m=s.dim[1];
    MADNESS_ASSERT(n==m);
    tensorT c, e;
    //s.gaxpy(0.5,transpose(s),0.5); // Ensure exact symmetry
    syev(s, &c, &e);
    for (int i=0; i<n; i++) {
        if (e(i) < -tol) {
            MADNESS_EXCEPTION("Matrix square root: negative eigenvalue",i);
        }
        else if (e(i) < tol) { // Ugh ..
            print("Matrix square root: Warning: small eigenvalue ", i, e(i));
            e(i) = tol;
        }
        e(i) = 1.0/sqrt(e(i));
    }
    for (int j=0; j<n; j++) {
        for (int i=0; i<n; i++) {
            c(j,i) *= e(i);
        }
    }
    return c;
}

template <typename T, int NDIM>
struct lbcost {
    double leaf_value;
    double parent_value;
    lbcost(double leaf_value=1.0, double parent_value=0.0) : leaf_value(leaf_value), parent_value(parent_value) {}
    double operator()(const Key<NDIM>& key, const FunctionNode<T,NDIM>& node) const {
        if (key.level() <= 1) {
            return 100.0*(leaf_value+parent_value);
        }
        else if (node.is_leaf()) {
            return leaf_value;
        }
        else {
            return parent_value;
        }
    }
};

// template <typename T, int NDIM>
// Cost lbcost(const Key<NDIM>& key, const FunctionNode<T,NDIM>& node) {
//   return 1;
// }

struct CalculationParameters {
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // !!!                                                                   !!!
    // !!! If you add more data don't forget to add them to serialize method !!!
    // !!!                                                                   !!!
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    // First list input parameters
    double charge;              ///< Total molecular charge
    double smear;               ///< Smearing parameter
    double econv;               ///< Energy convergence
    double dconv;               ///< Density convergence
    double L;                   ///< User coordinates box size
    double maxrotn;             ///< Step restriction used in autoshift algorithm
    int nvalpha;                ///< Number of alpha virtuals to compute
    int nvbeta;                 ///< Number of beta virtuals to compute
    int nopen;                  ///< Number of unpaired electrons = napha-nbeta
    int maxiter;                ///< Maximum number of iterations
    bool spin_restricted;       ///< True if spin restricted
    bool lda;                   ///< True if LDA (HF if false)
    int plotlo,plothi;          ///< Range of MOs to print (for both spins if polarized)
    bool plotdens;              ///< If true print the density at convergence
    bool plotcoul;              ///< If true plot the total coulomb potential at convergence
    bool localize;              ///< If true solve for localized orbitals
    unsigned int maxsub;        ///< Size of iterative subspace ... set to 0 or 1 to disable
    // Next list inferred parameters
    int nalpha;                 ///< Number of alpha spin electrons
    int nbeta;                  ///< Number of beta  spin electrons
    int nmo_alpha;              ///< Number of alpha spin molecular orbitals
    int nmo_beta;               ///< Number of beta  spin molecular orbitals
    double lo;                  ///< Smallest length scale we need to resolve

    template <typename Archive>
    void serialize(Archive& ar) {
        ar & charge & smear & econv & dconv & L & maxrotn & nvalpha & nvbeta & nopen & maxiter & spin_restricted & lda;
        ar & plotlo & plothi & plotdens & plotcoul & localize & maxsub;
        ar & nalpha & nbeta & nmo_alpha & nmo_beta & lo;
    }

    CalculationParameters()
            : charge(0.0)
            , smear(0.0)
            , econv(1e-5)
            , dconv(1e-4)
            , L(0.0)
            , maxrotn(0.25)
            , nvalpha(1)
            , nvbeta(1)
            , nopen(0)
            , maxiter(20)
            , spin_restricted(true)
            , lda(true)
            , plotlo(0)
            , plothi(-1)
            , plotdens(false)
            , plotcoul(false)
            , localize(true)
            , maxsub(8)
            , nalpha(0)
            , nbeta(0)
            , nmo_alpha(0)
            , nmo_beta(0)
            , lo(1e-10) {}

    void read_file(const std::string& filename) {
        std::ifstream f(filename.c_str());
        position_stream(f, "dft");
        string s;
        while (f >> s) {
            if (s == "end") {
                break;
            }
            else if (s == "charge") {
                f >> charge;
            }
            else if (s == "smear") {
                f >> smear;
            }
            else if (s == "econv") {
                f >> econv;
            }
            else if (s == "dconv") {
                f >> dconv;
            }
            else if (s == "L") {
                f >> L;
            }
            else if (s == "maxrotn") {
                f >> maxrotn;
            }
            else if (s == "nvalpha") {
                f >> nvalpha;
            }
            else if (s == "nvbeta") {
                f >> nvbeta;
            }
            else if (s == "nopen") {
                f >> nopen;
            }
            else if (s == "unrestricted") {
                spin_restricted = false;
            }
            else if (s == "restricted") {
                spin_restricted = true;
            }
            else if (s == "maxiter") {
                f >> maxiter;
            }
            else if (s == "lda") {
                lda = true;
            }
            else if (s == "hf") {
                lda = false;
            }
            else if (s == "plotmos") {
                f >> plotlo >> plothi;
            }
            else if (s == "plotdens") {
                plotdens = true;
            }
            else if (s == "plotcoul") {
                plotcoul = true;
            }
            else if (s == "canon") {
                localize = false;
            }
            else if (s == "local") {
                localize = true;
            }
            else if (s == "maxsub") {
                f >> maxsub;
                if (maxsub <= 0) maxsub = 1;
                if (maxsub > 20) maxsub = 20;
            }
            else {
                std::cout << "moldft: unrecognized input keyword " << s << std::endl;
                MADNESS_EXCEPTION("input error",0);
            }
            if (nopen != 0) spin_restricted = false;
        }
    }

    void set_molecular_info(const Molecule& molecule, const AtomicBasisSet& aobasis) {
        double z = molecule.total_nuclear_charge();
        int nelec = z - charge;
        if (fabs(nelec+charge-z) > 1e-6) {
            error("non-integer number of electrons?", nelec+charge-z);
        }
        nalpha = (nelec + nopen)/2;
        nbeta  = (nelec - nopen)/2;
        if (nalpha < 0) error("negative number of alpha electrons?", nalpha);
        if (nbeta < 0) error("negative number of beta electrons?", nbeta);
        if ((nalpha+nbeta) != nelec) error("nalpha+nbeta != nelec", nalpha+nbeta);
        nmo_alpha = nalpha + nvalpha;
        nmo_beta = nbeta + nvbeta;
        if (nalpha != nbeta) spin_restricted = false;

        // Ensure we have enough basis functions to guess the requested
        // number of states ... a minimal basis for a closed-shell atom
        // might not have any functions for virtuals.
        int nbf = aobasis.nbf(molecule);
        nmo_alpha = min(nbf,nmo_alpha);
        nmo_beta = min(nbf,nmo_beta);
        if (nalpha>nbf || nbeta>nbf) error("too few basis functions?", nbf);
        nvalpha = nmo_alpha - nalpha;
        nvbeta = nmo_beta - nbeta;

        // Unless overridden by the user use a cell big enough to
        // have exp(-sqrt(2*I)*r) decay to 1e-6 with I=1ev=0.037Eh
        // --> need 50 a.u. either side of the molecule

        if (L == 0.0) {
            L = molecule.bounding_cube() + 50.0;
        }

        lo = molecule.smallest_length_scale();
    }

    void print(World& world) const {
        //time_t t = time((time_t *) 0);
        //char *tmp = ctime(&t);
        //tmp[strlen(tmp)-1] = 0; // lose the trailing newline
        const char* calctype[2] = {"Hartree-Fock","LDA"};
        //madness::print(" date of calculation ", tmp);
        madness::print(" number of processes ", world.size());
        madness::print("        total charge ", charge);
        madness::print("            smearing ", smear);
        madness::print(" number of electrons ", nalpha, nbeta);
        madness::print("  number of orbitals ", nmo_alpha, nmo_beta);
        madness::print("     spin restricted ", spin_restricted);
        madness::print("  energy convergence ", econv);
        madness::print(" density convergence ", dconv);
        madness::print("    maximum rotation ", maxrotn);
        madness::print(" max krylov subspace ", maxsub);
        madness::print("    calculation type ", calctype[int(lda)]);
        if (localize) 
            madness::print("  localized orbitals ");
        else
            madness::print("  canonical orbitals ");
    }
};

struct Calculation {
    Molecule molecule;
    CalculationParameters param;
    AtomicBasisSet aobasis;
    functionT vnuc;
    functionT mask;
    vecfuncT amo, bmo;
    vector<int> aset, bset;
    vecfuncT ao;
    vector<int> at_to_bf, at_nbf;
    tensorT aocc, bocc;
    tensorT aeps, beps;
    poperatorT coulop;
    double vtol;
    Calculation(World & world, const char *filename)
    {
        if(world.rank() == 0){
            molecule.read_file(filename);
            param.read_file(filename);
            aobasis.read_file("sto-3g");
            molecule.orient();
            param.set_molecular_info(molecule, aobasis);
        }
        world.gop.broadcast_serializable(molecule, 0);
        world.gop.broadcast_serializable(param, 0);
        world.gop.broadcast_serializable(aobasis, 0);
        FunctionDefaults<3>::set_cubic_cell(-param.L, param.L);
        set_protocol(world, 1e-4);
    }

    void set_protocol(World & world, double thresh)
    {
        int k;
        if(thresh >= 1e-2)
            k = 4;

        else
            if(thresh >= 1e-4)
                k = 6;

            else
                if(thresh >= 1e-6)
                    k = 8;

                else
                    if(thresh >= 1e-8)
                        k = 10;

                    else
                        k = 12;

        FunctionDefaults<3>::set_k(k);
        FunctionDefaults<3>::set_thresh(thresh);
        FunctionDefaults<3>::set_refine(true);
        FunctionDefaults<3>::set_initial_level(2);
        FunctionDefaults<3>::set_truncate_mode(1);
        FunctionDefaults<3>::set_autorefine(false);
        FunctionDefaults<3>::set_apply_randomize(false);
        FunctionDefaults<3>::set_project_randomize(false);
        GaussianConvolution1DCache<double>::map.clear();
        double safety = 0.1;
        vtol = FunctionDefaults<3>::get_thresh() * safety;
        coulop = poperatorT(CoulombOperatorPtr<double>(world, FunctionDefaults<3>::get_k(), param.lo, thresh));
        mask = functionT(factoryT(world).f(mask3).initial_level(4).norefine());
        if(world.rank() == 0){
            print("\nSolving with thresh", thresh, "    k", FunctionDefaults<3>::get_k(), "   conv", max(thresh, param.dconv), "\n");
        }
    }

    void project(World & world)
    {
        reconstruct(world, amo);
        for(unsigned int i = 0;i < amo.size();i++){
            amo[i] = madness::project(amo[i], FunctionDefaults<3>::get_k(), FunctionDefaults<3>::get_thresh(), false);
            if (((i+1) % madness::VMRA_CHUNK_SIZE) == 0) world.gop.fence();
        }
        world.gop.fence();
        truncate(world, amo);
        normalize(world, amo);
        if(param.nbeta && !param.spin_restricted){
            reconstruct(world, bmo);
            for(unsigned int i = 0;i < bmo.size();i++){
                bmo[i] = madness::project(bmo[i], FunctionDefaults<3>::get_k(), FunctionDefaults<3>::get_thresh(), false);
                if (((i+1) % madness::VMRA_CHUNK_SIZE) == 0) world.gop.fence();
            }
            world.gop.fence();
            truncate(world, bmo);
            normalize(world, bmo);
        }

    }

    void make_nuclear_potential(World & world)
    {
        START_TIMER(world);
        vnuc = factoryT(world).functor(functorT(new MolecularPotentialFunctor(molecule))).thresh(vtol).truncate_on_project();
        vnuc.set_thresh(FunctionDefaults<3>::get_thresh());
        vnuc.reconstruct();
        END_TIMER(world, "Project vnuclear");
    }

    void project_ao_basis(World & world)
    {
        // Make at_to_bf, at_nbf ... map from atom to first bf on atom, and nbf/atom
        aobasis.atoms_to_bfn(molecule, at_to_bf, at_nbf);

        START_TIMER(world);
        ao = vecfuncT(aobasis.nbf(molecule));
        Level initial_level = 2;
        for(int i = 0;i < aobasis.nbf(molecule);i++){
            functorT aofunc(new AtomicBasisFunctor(aobasis.get_atomic_basis_function(molecule, i)));
            ao[i] = factoryT(world).functor(aofunc).initial_level(initial_level).truncate_on_project().nofence();
            if (((i+1) % madness::VMRA_CHUNK_SIZE) == 0) world.gop.fence();
        }
        world.gop.fence();
        vector<double> norms;
        while(1){
            norms = norm2(world, ao);
            initial_level += 2;
            if(initial_level >= 11)
                throw "project_ao_basis: projection failed?";

            int nredone = 0;
            for(int i = 0;i < aobasis.nbf(molecule);i++){
                if(norms[i] < 0.999){
                    nredone++;
                    functorT aofunc(new AtomicBasisFunctor(aobasis.get_atomic_basis_function(molecule, i)));
                    ao[i] = factoryT(world).functor(aofunc).initial_level(6).truncate_on_project().nofence();
                }
            }

            world.gop.fence();
            if(nredone == 0)
                break;

        }
        truncate(world, ao);
        for(int i = 0;i < aobasis.nbf(molecule);i++){
            if(world.rank() == 0 && fabs(norms[i] - 1.0) > 1e-3)
                print(i, " bad ao norm?", norms[i]);

            norms[i] = 1.0 / norms[i];
        }
        scale(world, ao, norms);
        END_TIMER(world, "project ao basis");
    }

    double PM_q(const tensorT & S, const tensorT & C, int i, int j, int lo, int nbf)
    {
        double qij = 0.0;
        if (nbf == 1) { // H atom in STO-3G ... often lots of these!
            qij = C(i,lo)*S(0,0)*C(j,lo);
        }
        else {
            for(int mu = 0;mu < nbf;mu++){
                double Smuj = 0.0;
                for(int nu = 0;nu < nbf;nu++){
                    Smuj += S(mu, nu) * C(j, nu + lo);
                }
                qij += C(i, mu + lo) * Smuj;
            }
        }

        return qij;
    }

    void localize_PM_task_kernel(tensorT & Q, vector<tensorT> & Svec, tensorT & C, 
                                 const bool & doprint, const vector<int> & set, 
                                 const double thetamax, tensorT & U, const double thresh)
    {
        long nmo = C.dim[0];
        long nao = C.dim[1];
        long natom = molecule.natom();

        for(long i = 0;i < nmo;i++){
            for(long a = 0;a < natom;a++){
                Q(i, a) = PM_q(Svec[a], C, i, i, at_to_bf[a], at_nbf[a]);
            }
        }

        double tol = 0.1;
        long ndone = 0;
        bool converged = false;
        for(long iter = 0;iter < 100;iter++){
            double sum = 0.0;
            for(long i = 0;i < nmo;i++){
                for(long a = 0;a < natom;a++){
                    double qiia = Q(i, a);
                    sum += qiia * qiia;
                }
            }

            long ndone_iter = 0;
            double maxtheta = 0.0;
            // if(doprint)
            //     printf("iteration %ld sum=%.4f ndone=%ld tol=%.2e\n", iter, sum, ndone, tol);

            for(long i = 0;i < nmo;i++){
                for(long j = 0;j < i;j++){
                    if(set[i] == set[j]){
                        double ovij = 0.0;
                        for(long a = 0;a < natom;a++)
                            ovij += Q(i, a) * Q(j, a);

                        if(fabs(ovij) > tol * tol){
                            double aij = 0.0;
                            double bij = 0.0;
                            for(long a = 0;a < natom;a++){
                                double qiia = Q(i, a);
                                double qija = PM_q(Svec[a], C, i, j, at_to_bf[a], at_nbf[a]);
                                double qjja = Q(j, a);
                                double d = qiia - qjja;
                                aij += qija * qija - 0.25 * d * d;
                                bij += qija * d;
                            }
                            double theta = 0.25 * acos(-aij / sqrt(aij * aij + bij * bij));
                            if(bij > 0.0)
                                theta = -theta;

                            if(theta > thetamax)
                                theta = thetamax;

                            else
                                if(theta < -thetamax)
                                    theta = -thetamax;


                            maxtheta = max(fabs(theta), maxtheta);
                            if(fabs(theta) >= tol){
                                ndone_iter++;
                                double c = cos(theta);
                                double s = sin(theta);
                                drot(nao, &C(i, 0), &C(j, 0), s, c, 1);
                                drot(nmo, &U(i, 0), &U(j, 0), s, c, 1);
                                for(long a = 0;a < natom;a++){
                                    Q(i, a) = PM_q(Svec[a], C, i, i, at_to_bf[a], at_nbf[a]);
                                    Q(j, a) = PM_q(Svec[a], C, j, j, at_to_bf[a], at_nbf[a]);
                                }
                            }

                        }

                    }

                }

            }

            ndone += ndone_iter;
            if(ndone_iter == 0 && tol == thresh){
                if(doprint)
                    print("PM localization converged in", ndone,"steps");

                converged = true;
                break;
            }
            tol = max(0.1 * min(maxtheta, tol), thresh);
        }

    }

    tensorT localize_PM(World & world, const vecfuncT & mo, const vector<int> & set, const double thresh = 1e-9, const double thetamax = 0.5, const bool randomize = true, const bool doprint = true)
    {
        START_TIMER(world);
        long nmo = mo.size();
        long natom = molecule.natom();

        tensorT S = matrix_inner(world, ao, ao, true);
        vector<tensorT> Svec(natom);
        for(long a = 0;a < natom;a++){
            Slice as(at_to_bf[a], at_to_bf[a] + at_nbf[a] - 1);
            Svec[a] = copy(S(as, as));
        }
        S = tensorT();
        tensorT C = matrix_inner(world, mo, ao);
        tensorT U(nmo, nmo);
        tensorT Q(nmo, natom);
        if(world.rank() == 0){
            for(long i = 0;i < nmo;i++)
                U(i, i) = 1.0;

            localize_PM_task_kernel(Q, Svec, C, doprint, set, thetamax, U, thresh);
            U = transpose(U);
        }
        world.gop.broadcast(U.ptr(), U.size, 0);
        END_TIMER(world, "Pipek-Mezy localize");
        return U;
    }

    void analyze_vectors(World & world, const vecfuncT & mo, const tensorT & occ = tensorT(), const tensorT & energy = tensorT(), const vector<int> & set = vector<int>())
    {
        tensorT Saomo = matrix_inner(world, ao, mo);
        tensorT Saoao = matrix_inner(world, ao, ao, true);
        int nmo = mo.size();
        tensorT rsq, dip(3, nmo);
        {
            functionT frsq = factoryT(world).f(rsquared).initial_level(4);
            rsq = inner(world, mo, mul_sparse(world, frsq, mo, vtol));
            for(int axis = 0;axis < 3;axis++){
                functionT fdip = factoryT(world).functor(functorT(new DipoleFunctor(axis))).initial_level(4);
                dip(axis, _) = inner(world, mo, mul_sparse(world, fdip, mo, vtol));
                for(int i = 0;i < nmo;i++)
                    rsq(i) -= dip(axis, i) * dip(axis, i);

            }
        }
        if(world.rank() == 0){
            tensorT C;
            gesv(Saoao, Saomo, &C);
            C = transpose(C);
            long nmo = mo.size();
            for(long i = 0;i < nmo;i++){
                printf("  MO%4ld : ", i);
                if(set.size())
                    printf("set=%d : ", set[i]);

                if(occ.size)
                    printf("occ=%.2f : ", occ(i));

                if(energy.size)
                    printf("energy=%11.6f : ", energy(i));

                printf("center=(%.2f,%.2f,%.2f) : radius=%.2f\n", dip(0, i), dip(1, i), dip(2, i), sqrt(rsq(i)));
                aobasis.print_anal(molecule, C(i, _));
            }
        }

    }

    inline double DIP(const tensorT & dip, int i, int j, int k, int l)
    {
        return dip(i, j, 0) * dip(k, l, 0) + dip(i, j, 1) * dip(k, l, 1) + dip(i, j, 2) * dip(k, l, 2);
    }

    tensorT localize_boys(World & world, const vecfuncT & mo, const double thresh = 1e-9, const double thetamax = 0.5, const bool randomize = true)
    {
        START_TIMER(world);
        const bool doprint = false;
        long nmo = mo.size();
        tensorT dip(nmo, nmo, 3);
        for(int axis = 0;axis < 3;axis++){
            functionT fdip = factoryT(world).functor(functorT(new DipoleFunctor(axis))).initial_level(4);
            dip(_, _, axis) = matrix_inner(world, mo, mul_sparse(world, fdip, mo, vtol), true);
        }
        tensorT U(nmo, nmo);
        if(world.rank() == 0){
            for(long i = 0;i < nmo;i++)
                U(i, i) = 1.0;

            double tol = thetamax;
            long ndone = 0;
            bool converged = false;
            for(long iter = 0;iter < 300;iter++){
                double sum = 0.0;
                for(long i = 0;i < nmo;i++){
                    sum += DIP(dip, i, i, i, i);
                }
                long ndone_iter = 0;
                double maxtheta = 0.0;
                if(doprint)
                    printf("iteration %ld sum=%.4f ndone=%ld tol=%.2e\n", iter, sum, ndone, tol);

                for(long i = 0;i < nmo;i++){
                    for(long j = 0;j < i;j++){
                        double g = DIP(dip, i, j, j, j) - DIP(dip, i, j, i, i);
                        double h = 4.0 * DIP(dip, i, j, i, j) + 2.0 * DIP(dip, i, i, j, j) - DIP(dip, i, i, i, i) - DIP(dip, j, j, j, j);
                        double sij = DIP(dip, i, j, i, j);
                        bool doit = false;
                        if(h >= 0.0){
                            doit = true;
                            if(doprint)
                                print("             forcing negative h", i, j, h);

                            h = -1.0;
                        }
                        double theta = -g / h;
                        maxtheta = max(abs(theta), maxtheta);
                        if(fabs(theta) > thetamax){
                            doit = true;
                            if(doprint)
                                print("             restricting", i, j);

                            if(g < 0)
                                theta = -thetamax;

                            else
                                theta = thetamax * 0.8;

                        }
                        bool randomized = false;
                        if(randomize && iter == 0 && sij > 0.01 && fabs(theta) < 0.01){
                            randomized = true;
                            if(doprint)
                                print("             randomizing", i, j);

                            theta += (RandomValue<double>() - 0.5);
                        }
                        if(fabs(theta) >= tol || randomized || doit){
                            ndone_iter++;
                            if(doprint)
                                print("     rotating", i, j, theta);

                            double c = cos(theta);
                            double s = sin(theta);
                            drot3(nmo, &dip(i, 0, 0), &dip(j, 0, 0), s, c, 1);
                            drot3(nmo, &dip(0, i, 0), &dip(0, j, 0), s, c, nmo);
                            drot(nmo, &U(i, 0), &U(j, 0), s, c, 1);
                        }
                    }

                }

                ndone += ndone_iter;
                if(ndone_iter == 0 && tol == thresh){
                    if(doprint)
                        print("Boys localization converged in", ndone,"steps");

                    converged = true;
                    break;
                }
                tol = max(0.1 * maxtheta, thresh);
            }

            if(!converged){
                print("warning: boys localization did not fully converge: ", ndone);
            }
            U = transpose(U);
        }

        world.gop.broadcast(U.ptr(), U.size, 0);
        END_TIMER(world, "Boys localize");
        return U;
    }

    tensorT kinetic_energy_matrix(World & world, const vecfuncT & v)
    {
        reconstruct(world, v);
        int n = v.size();
        tensorT r(n, n);
        for(int axis = 0;axis < 3;axis++){
            vecfuncT dv = diff(world, v, axis);
            r += matrix_inner(world, dv, dv, true);
            dv.clear();
        }
        return r.scale(0.5);
    }

    struct GuessDensity : public FunctionFunctorInterface<double,3>
    {
        const Molecule & molecule;
        const AtomicBasisSet & aobasis;
        double operator ()(const coordT & x) const
        {
            return aobasis.eval_guess_density(molecule, x[0], x[1], x[2]);
        }

        GuessDensity(const Molecule & molecule, const AtomicBasisSet & aobasis)
        :molecule(molecule), aobasis(aobasis)
        {
        }

    };

    void initial_guess(World & world)
    {
        START_TIMER(world);
        functionT rho = factoryT(world).functor(functorT(new GuessDensity(molecule, aobasis))).truncate_on_project();
        END_TIMER(world, "guess density");
        double nel = rho.trace();
        if(world.rank() == 0)
            print("guess dens trace", nel);

        if(world.size() > 1){
            START_TIMER(world);
            LoadBalanceDeux<3> lb(world);
            lb.add_tree(vnuc, lbcost<double,3>(1.0, 0.0), true);
            lb.add_tree(rho, lbcost<double,3>(1.0, 1.0), false);
            FunctionDefaults<3>::set_pmap(lb.load_balance(6.0));
            rho = copy(rho, FunctionDefaults<3>::get_pmap(), false);
            vnuc = copy(vnuc, FunctionDefaults<3>::get_pmap(), false);
            world.gop.fence();
            END_TIMER(world, "guess loadbal");
        }
        functionT vlocal;
        if(param.nalpha + param.nbeta > 1){
            START_TIMER(world);
            vlocal = vnuc + apply(*coulop, rho);
            END_TIMER(world, "guess Coulomb potn");
            bool save = param.spin_restricted;
            param.spin_restricted = true;
            rho.scale(0.5);
            vlocal = vlocal + make_lda_potential(world, rho, rho, functionT(), functionT());
            vlocal.truncate();
            param.spin_restricted = save;
        }else{
            vlocal = vnuc;
        }
        rho.clear();
        vlocal.reconstruct();
        if(world.size() > 1){
            LoadBalanceDeux<3> lb(world);
            lb.add_tree(vnuc, lbcost<double,3>(1.0, 1.0), false);
            for(unsigned int i = 0;i < ao.size();i++){
                lb.add_tree(ao[i], lbcost<double,3>(1.0, 1.0), false);
            }
            FunctionDefaults<3>::set_pmap(lb.load_balance(6.0));
            vnuc = copy(vnuc, FunctionDefaults<3>::get_pmap(), false);
            vlocal = copy(vlocal, FunctionDefaults<3>::get_pmap(), false);
            for(unsigned int i = 0;i < ao.size();i++){
                ao[i] = copy(ao[i], FunctionDefaults<3>::get_pmap(), false);
            }
            world.gop.fence();
        }

        START_TIMER(world);
        tensorT overlap = matrix_inner(world, ao, ao, true);
        END_TIMER(world, "make overlap");
        START_TIMER(world);
        tensorT kinetic = kinetic_energy_matrix(world, ao);
        END_TIMER(world, "make KE matrix");
        reconstruct(world, ao);
        vlocal.reconstruct();
        START_TIMER(world);
        vecfuncT vpsi = mul_sparse(world, vlocal, ao, vtol);
        END_TIMER(world, "make V*psi");
        START_TIMER(world);
        compress(world, vpsi);
        END_TIMER(world, "Compressing Vpsi");
        START_TIMER(world);
        truncate(world, vpsi);
        END_TIMER(world, "Truncating vpsi");
        START_TIMER(world);
        compress(world, ao);
        END_TIMER(world, "Compressing AO");
        START_TIMER(world);
        tensorT potential = matrix_inner(world, vpsi, ao, true);
        END_TIMER(world, "make PE matrix");
        vpsi.clear();
        tensorT fock = kinetic + potential;
        fock = 0.5 * (fock + transpose(fock));
        tensorT c, e;
        sygv(fock, overlap, 1, &c, &e);
        world.gop.broadcast(c.ptr(), c.size, 0);
        world.gop.broadcast(e.ptr(), e.size, 0);
        if(world.rank() == 0){
            print("initial eigenvalues");
            print(e);
        }
        compress(world, ao);
        START_TIMER(world);
        amo = transform(world, ao, c(_, Slice(0, param.nmo_alpha - 1)), 0.0, true);
        END_TIMER(world, "transform initial MO");
        truncate(world, amo);
        normalize(world, amo);
        aeps = e(Slice(0, param.nmo_alpha - 1));
        aocc = tensorT(param.nmo_alpha);
        for(int i = 0;i < param.nalpha;i++)
            aocc[i] = 1.0;

        aset = vector<int>(param.nmo_alpha);
        aset[0] = 0;
        if(world.rank() == 0)
            cout << "alpha set " << 0 << " " << 0 << "-";

        for(int i = 1;i < param.nmo_alpha;i++){
            aset[i] = aset[i - 1];
            if(aeps[i] - aeps[i - 1] > 1.5 || aocc[i] != 1.0){
                aset[i]++;
                if(world.rank() == 0){
                    cout << i - 1 << endl;
                    cout << "alpha set " << aset[i] << " " << i << "-";
                }
            }

        }

        if(world.rank() == 0)
            cout << param.nmo_alpha - 1 << endl;

        if(param.nbeta && !param.spin_restricted){
            bmo = transform(world, ao, c(_, Slice(0, param.nmo_beta - 1)), 0.0, true);
            truncate(world, bmo);
            normalize(world, bmo);
            beps = e(Slice(0, param.nmo_beta - 1));
            bocc = tensorT(param.nmo_beta);
            for(int i = 0;i < param.nbeta;i++)
                bocc[i] = 1.0;

            bset = vector<int>(param.nmo_beta);
            bset[0] = 0;
            if(world.rank() == 0)
                cout << " beta set " << 0 << " " << 0 << "-";

            for(int i = 1;i < param.nmo_beta;i++){
                bset[i] = bset[i - 1];
                if(beps[i] - beps[i - 1] > 1.5 || bocc[i] != 1.0){
                    bset[i]++;
                    if(world.rank() == 0){
                        cout << i - 1 << endl;
                        cout << " beta set " << bset[i] << " " << i << "-";
                    }
                }

            }

            if(world.rank() == 0)
                cout << param.nmo_beta - 1 << endl;

        }
    }

    void initial_load_bal(World & world)
    {
        LoadBalanceDeux<3> lb(world);
        lb.add_tree(vnuc, lbcost<double,3>(1.0, 0.0));
        FunctionDefaults<3>::set_pmap(lb.load_balance(6.0));
        world.gop.fence();
    }

    functionT make_density(World & world, const tensorT & occ, const vecfuncT & v)
    {
        vecfuncT vsq = square(world, v);
        compress(world, vsq);
        functionT rho = factoryT(world);
        rho.compress();
        for(unsigned int i = 0;i < vsq.size();i++){
            if(occ[i])
                rho.gaxpy(1.0, vsq[i], occ[i], false);

        }
        world.gop.fence();
        vsq.clear();
        return rho;
    }

    vector<poperatorT> make_bsh_operators(World & world, const tensorT & evals)
    {
        int nmo = evals.dim[0];
        vector<poperatorT> ops(nmo);
        int k = FunctionDefaults<3>::get_k();
        double tol = FunctionDefaults<3>::get_thresh();
        for(int i = 0;i < nmo;i++){
            double eps = evals(i);
            if(eps > 0){
                if(world.rank() == 0){
                    print("bsh: warning: positive eigenvalue", i, eps);
                }
                eps = -0.1;
            }

            ops[i] = poperatorT(BSHOperatorPtr3D<double>(world, sqrt(-2.0 * eps), k, param.lo, tol));
        }

        return ops;
    }

    vecfuncT apply_hf_exchange(World & world, const tensorT & occ, const vecfuncT & psi, const vecfuncT & f)
    {
        int nocc = psi.size();
        int nf = f.size();
        double tol = FunctionDefaults<3>::get_thresh();
        vecfuncT Kf = zero_functions<double,3>(world, nf);
        compress(world, Kf);
        reconstruct(world, psi);
        for(int i = 0;i < nocc;i++){
            if(occ[i] > 0.0){
                vecfuncT psif = mul_sparse(world, psi[i], f, vtol);
                truncate(world, psif);
                psif = apply(world, *coulop, psif);
                truncate(world, psif);
                psif = mul_sparse(world, psi[i], psif, vtol);
                gaxpy(world, 1.0, Kf, occ[i], psif);
            }
        }

        truncate(world, Kf, vtol);
        return Kf;
    }

    static double munge(double r)
    {
        if(r < 1e-12)
            r = 1e-12;

        return r;
    }
    static void ldaop(const Key<3> & key, tensorT & t)
    {
        UNARY_OPTIMIZED_ITERATOR(double, t, double r=munge(2.0* *_p0); double q; double dq1; double dq2;x_rks_s__(&r, &q, &dq1);c_rks_vwn5__(&r, &q, &dq2); *_p0 = dq1+dq2);
    }

    static void ldaeop(const Key<3> & key, tensorT & t)
    {
        UNARY_OPTIMIZED_ITERATOR(double, t, double r=munge(2.0* *_p0); double q1; double q2; double dq;x_rks_s__(&r, &q1, &dq);c_rks_vwn5__(&r, &q2, &dq); *_p0 = q1+q2);
    }

    functionT make_lda_potential(World & world, const functionT & arho, const functionT & brho, const functionT & adelrhosq, const functionT & bdelrhosq)
    {
        MADNESS_ASSERT(param.spin_restricted);
        functionT vlda = copy(arho);
        vlda.reconstruct();
        vlda.unaryop(&ldaop);
        return vlda;
    }

    double make_lda_energy(World & world, const functionT & arho, const functionT & brho, const functionT & adelrhosq, const functionT & bdelrhosq)
    {
        MADNESS_ASSERT(param.spin_restricted);
        functionT vlda = copy(arho);
        vlda.reconstruct();
        vlda.unaryop(&ldaeop);
        return vlda.trace();
    }

    vecfuncT apply_potential(World & world, const tensorT & occ, const vecfuncT & amo, const functionT & arho, const functionT & brho, const functionT & adelrhosq, const functionT & bdelrhosq, const functionT & vlocal, double & exc)
    {
        functionT vloc = vlocal;
        if(param.lda){
            START_TIMER(world);
            exc = make_lda_energy(world, arho, brho, adelrhosq, bdelrhosq);
            vloc = vloc + make_lda_potential(world, arho, brho, adelrhosq, bdelrhosq);
            END_TIMER(world, "LDA potential");
        }
        START_TIMER(world);
        vecfuncT Vpsi = mul_sparse(world, vloc, amo, vtol);
        END_TIMER(world, "V*psi");
        if(!param.lda){
            START_TIMER(world);
            vecfuncT Kamo = apply_hf_exchange(world, occ, amo, amo);
            tensorT excv = inner(world, Kamo, amo);
            exc = 0.0;
            for(unsigned long i = 0;i < amo.size();i++){
                exc -= 0.5 * excv[i] * occ[i];
            }
            gaxpy(world, 1.0, Vpsi, -1.0, Kamo);
            Kamo.clear();
            END_TIMER(world, "HF exchange");
        }

        START_TIMER(world);
        truncate(world, Vpsi);
        END_TIMER(world, "Truncate Vpsi");
        world.gop.fence();
        return Vpsi;
    }

    tensorT derivatives(World & world, const functionT & rho)
    {
        vecfuncT dv(molecule.natom() * 3);
        for(int atom = 0;atom < molecule.natom();atom++){
            for(int axis = 0;axis < 3;axis++){
                functorT func(new MolecularDerivativeFunctor(molecule, atom, axis));
                dv[atom * 3 + axis] = functionT(factoryT(world).functor(func).nofence().truncate_on_project());
            }
        }

        world.gop.fence();
        tensorT r = inner(world, rho, dv);
        dv.clear();
        world.gop.fence();
        for(int atom = 0;atom < molecule.natom();atom++){
            for(int axis = 0;axis < 3;axis++){
                r[atom * 3 + axis] += molecule.nuclear_repulsion_derivative(atom, axis);
            }
        }

        return r;
    }

    void vector_stats(const vector<double> & v, double & rms, double & maxabsval)
    {
        rms = 0.0;
        maxabsval = v[0];
        for(unsigned int i = 0;i < v.size();i++){
            rms += v[i] * v[i];
            maxabsval = max(maxabsval, abs(v[i]));
        }
        rms = sqrt(rms / v.size());
    }

    vecfuncT compute_residual(World & world, tensorT & occ, tensorT & fock, const vecfuncT & psi, vecfuncT & Vpsi, double & err)
    {
        double trantol = vtol / min(30.0, double(psi.size()));
        int nmo = psi.size();

        tensorT eps(nmo);
        for(int i = 0;i < nmo;i++){
            eps(i) = min(-0.05, fock(i, i));
            fock(i, i) -= eps(i);
        }
        vecfuncT fpsi = transform(world, psi, fock, trantol, true);

        for(int i = 0;i < nmo;i++){ // Undo the damage
            fock(i, i) += eps(i);
        }

        gaxpy(world, 1.0, Vpsi, -1.0, fpsi);
        fpsi.clear();
        vector<double> fac(nmo, -2.0);
        scale(world, Vpsi, fac);
        vector<poperatorT> ops = make_bsh_operators(world, eps);
        set_thresh(world, Vpsi, FunctionDefaults<3>::get_thresh());
        if(world.rank() == 0)
            cout << "entering apply\n";

        START_TIMER(world);
        vecfuncT new_psi = apply(world, ops, Vpsi);
        END_TIMER(world, "Apply BSH");
        ops.clear();
        Vpsi.clear();
        world.gop.fence();
        START_TIMER(world);
        truncate(world, new_psi);
        END_TIMER(world, "Truncate new psi");
        vecfuncT r = sub(world, psi, new_psi);
        vector<double> rnorm = norm2(world, r);
        double rms, maxval;
        vector_stats(rnorm, rms, maxval);
        err = maxval;
        if(world.rank() == 0)
            print("BSH residual: rms", rms, "   max", maxval);

        return r;
    }

    tensorT make_fock_matrix(World & world, const vecfuncT & psi, const vecfuncT & Vpsi, const tensorT & occ, double & ekinetic)
    {
        START_TIMER(world);
        tensorT pe = matrix_inner(world, Vpsi, psi, true);
        END_TIMER(world, "PE matrix");
        START_TIMER(world);
        tensorT ke = kinetic_energy_matrix(world, psi);
        END_TIMER(world, "KE matrix");
        int nocc = occ.size;
        ekinetic = 0.0;
        for(int i = 0;i < nocc;i++){
            ekinetic += occ[i] * ke(i, i);
        }
        ke += pe;
        pe = tensorT();
        ke.gaxpy(0.5, transpose(ke), 0.5);
        return ke;
    }

    tensorT matrix_exponential(const tensorT& A) {
        const double tol = 1e-13;
        MADNESS_ASSERT(A.dim[0] == A.dim[1]);

        // Scale A by a power of 2 until it is "small"
        double anorm = A.normf();
        int n = 0;
        double scale = 1.0;
        while (anorm*scale > 0.1) {
            n++;
            scale *= 0.5;
        }
        tensorT B = scale*A;    // B = A*2^-n

        // Compute exp(B) using Taylor series
        tensorT expB = tensorT(2, B.dim);
        for (int i=0; i<expB.dim[0]; i++) expB(i,i) = 1.0;

        int k = 1;
        tensorT term = B;
        while (term.normf() > tol) {
            expB += term;
            term = inner(term,B);
            k++;
            term.scale(1.0/k);
        }

        // Repeatedly square to recover exp(A)
        while (n--) {
            expB = inner(expB,expB);
        }

        return expB;
    }

    void diag_fock_matrix(World & world, tensorT& fock, vecfuncT & psi, vecfuncT & Vpsi, tensorT & evals, double thresh)
    {
        long nmo = psi.size();
        tensorT overlap = matrix_inner(world, psi, psi, true);
        START_TIMER(world);
        tensorT U;

        sygv(fock, overlap, 1, &U, &evals);
        END_TIMER(world, "Diagonalization");

        // Fix phases.
        long j;
        for (long i=0; i<nmo; i++) {
            U(_,i).absmax(&j);
            if (U(j,i) < 0) U(_,i).scale(-1.0);
        }

        // Within blocks with the same occupation number attempt to
        // keep orbitals in the same order (to avoid confusing the
        // non-linear solver).  Have to run the reordering multiple
        // times to handle multiple degeneracies.
        for (int pass=0; pass<5; pass++) {
            for (long i=0; i<nmo; i++) {
                U(_,i).absmax(&j);
                if (i != j) {
                    tensorT tmp = copy(U(_,i));
                    U(_,i) = U(_,j);
                    U(_,j) = tmp;
                    swap(evals[i],evals[j]);
                }
            }
        }

        // Rotations between effectively degenerate states confound
        // the non-linear equation solver ... undo these rotations
        long ilo = 0; // first element of cluster
        while (ilo < nmo-1) {
            long ihi = ilo;
            while (fabs(evals[ilo]-evals[ihi+1]) < thresh*10.0*max(fabs(evals[ilo]),1.0)) {
                ihi++;
                if (ihi == nmo-1) break;
            }
            long nclus = ihi - ilo + 1;
            if (nclus > 1) {
                print("   found cluster", ilo, ihi);
                tensorT q = copy(U(Slice(ilo,ihi),Slice(ilo,ihi)));
                //print(q);
                // Special code just for nclus=2
                // double c = 0.5*(q(0,0) + q(1,1));
                // double s = 0.5*(q(0,1) - q(1,0));
                // double r = sqrt(c*c + s*s);
                // c /= r;
                // s /= r;
                // q(0,0) = q(1,1) = c;
                // q(0,1) = -s;
                // q(1,0) = s;

                // Iteratively construct unitary rotation by
                // exponentiating the antisymmetric part of the matrix
                // ... is quadratically convergent so just do 3
                // iterations
                tensorT rot = matrix_exponential(-0.5*(q - transpose(q)));
                q = inner(q,rot);
                tensorT rot2 = matrix_exponential(-0.5*(q - transpose(q)));
                q = inner(q,rot2);
                tensorT rot3 = matrix_exponential(-0.5*(q - transpose(q)));
                q = inner(rot,inner(rot2,rot3));
                U(_,Slice(ilo,ihi)) = inner(U(_,Slice(ilo,ihi)),q);
            }    
            ilo = ihi+1;
        }

        print("Fock");
        print(fock);
        print("Evec");
        print(U);;
        print("Eval");
        print(evals);

        world.gop.broadcast(U.ptr(), U.size, 0);
        world.gop.broadcast(evals.ptr(), evals.size, 0);

        fock = 0;
        for (unsigned int i=0; i<psi.size(); i++) fock(i,i) = evals(i);

        Vpsi = transform(world, Vpsi, U, vtol / min(30.0, double(psi.size())), false);
        psi = transform(world, psi, U, FunctionDefaults<3>::get_thresh() / min(30.0, double(psi.size())), true);

        if(world.rank() == 0)
            printf("transformed psi and Vpsi at %.2fs\n", wall_time());

        truncate(world, psi);
        if(world.rank() == 0)
            printf("truncated psi at %.2fs\n", wall_time());

        normalize(world, psi);
        if(world.rank() == 0)
            printf("normalized psi at %.2fs\n", wall_time());
    }

    void loadbal(World & world, functionT & arho, functionT & brho, functionT & arho_old, functionT & brho_old, subspaceT & subspace)
    {
        if(world.size() == 1)
            return;

        LoadBalanceDeux<3> lb(world);
        lb.add_tree(vnuc, lbcost<double,3>(1.0, 0.0), false);
        lb.add_tree(arho, lbcost<double,3>(1.0, 1.0), false);
        for(unsigned int i = 0;i < amo.size();i++){
            lb.add_tree(amo[i], lbcost<double,3>(1.0, 1.0), false);
        }
        if(param.nbeta && !param.spin_restricted){
            lb.add_tree(brho, lbcost<double,3>(1.0, 1.0), false);
            for(unsigned int i = 0;i < bmo.size();i++){
                lb.add_tree(bmo[i], lbcost<double,3>(1.0, 1.0), false);
            }
        }

        FunctionDefaults<3>::set_pmap(lb.load_balance(6.0));
        vnuc = copy(vnuc, FunctionDefaults<3>::get_pmap(), false);
        arho = copy(arho, FunctionDefaults<3>::get_pmap(), false);
        if(arho_old.is_initialized())
            arho_old = copy(arho_old, FunctionDefaults<3>::get_pmap(), false);

        for(unsigned int i = 0;i < ao.size();i++){
            ao[i] = copy(ao[i], FunctionDefaults<3>::get_pmap(), false);
        }
        for(unsigned int i = 0;i < amo.size();i++){
            amo[i] = copy(amo[i], FunctionDefaults<3>::get_pmap(), false);
        }
        if(param.nbeta && !param.spin_restricted){
            brho = copy(brho, FunctionDefaults<3>::get_pmap(), false);
            if(brho_old.is_initialized())
                brho_old = copy(brho_old, FunctionDefaults<3>::get_pmap(), false);

            for(unsigned int i = 0;i < bmo.size();i++){
                bmo[i] = copy(bmo[i], FunctionDefaults<3>::get_pmap(), false);
            }
        }

        for(unsigned int i = 0;i < subspace.size();i++){
            vecfuncT & v = subspace[i].first;
            vecfuncT & r = subspace[i].second;
            for(unsigned int j = 0;j < v.size();j++){
                v[j] = copy(v[j], FunctionDefaults<3>::get_pmap(), false);
                r[j] = copy(r[j], FunctionDefaults<3>::get_pmap(), false);
            }
        }

        world.gop.fence();
    }

    void update_subspace(World & world, 
                         vecfuncT & Vpsia, vecfuncT & Vpsib, 
                         tensorT & focka, tensorT & fockb, 
                         subspaceT & subspace, tensorT & Q, 
                         double & bsh_residual, double & update_residual)
    {
        double aerr = 0.0, berr = 0.0;
        vecfuncT vm = amo;
        vecfuncT rm = compute_residual(world, aocc, focka, amo, Vpsia, aerr);
        if(param.nbeta && !param.spin_restricted){
            vecfuncT br = compute_residual(world, bocc, fockb, bmo, Vpsib, berr);
            vm.insert(vm.end(), bmo.begin(), bmo.end());
            rm.insert(rm.end(), br.begin(), br.end());
        }
        bsh_residual = max(aerr, berr);
        world.gop.broadcast(bsh_residual, 0);
        compress(world, vm, false);
        compress(world, rm, false);
        world.gop.fence();
        subspace.push_back(pairvecfuncT(vm, rm));
        int m = subspace.size();
        tensorT ms(m);
        tensorT sm(m);
        for(int s = 0;s < m;s++){
            const vecfuncT & vs = subspace[s].first;
            const vecfuncT & rs = subspace[s].second;
            for(unsigned int i = 0;i < vm.size();i++){
                ms[s] += vm[i].inner_local(rs[i]);
                sm[s] += vs[i].inner_local(rm[i]);
            }
        }

        world.gop.sum(ms.ptr(), m);
        world.gop.sum(sm.ptr(), m);
        tensorT newQ(m, m);
        if(m > 1)
            newQ(Slice(0, -2), Slice(0, -2)) = Q;

        newQ(m - 1, _) = ms;
        newQ(_, m - 1) = sm;
        Q = newQ;
        tensorT c;
        if(world.rank() == 0){
            double rcond = 1e-12;
            while(1){
                c = KAIN(Q, rcond);
                if(abs(c[m - 1]) < 3.0){
                    break;
                } else  if(rcond < 0.01){
                    print("Increasing subspace singular value threshold ", c[m - 1], rcond);
                    rcond *= 100;
                } else {
                    print("Forcing full step due to subspace malfunction");
                    c = 0.0;
                    c[m - 1] = 1.0;
                    break;
                }
            }
        }

        world.gop.broadcast_serializable(c, 0);
        if(world.rank() == 0){
            print("Subspace solution", c);
        }
        START_TIMER(world);
        vecfuncT amo_new = zero_functions<double,3>(world, amo.size());
        vecfuncT bmo_new = zero_functions<double,3>(world, bmo.size());
        compress(world, amo_new, false);
        compress(world, bmo_new, false);
        world.gop.fence();
        for(unsigned int m = 0;m < subspace.size();m++){
            const vecfuncT & vm = subspace[m].first;
            const vecfuncT & rm = subspace[m].second;
            const vecfuncT vma(vm.begin(), vm.begin() + amo.size());
            const vecfuncT rma(rm.begin(), rm.begin() + amo.size());
            const vecfuncT vmb(vm.end() - bmo.size(), vm.end());
            const vecfuncT rmb(rm.end() - bmo.size(), rm.end());
            gaxpy(world, 1.0, amo_new, c(m), vma, false);
            gaxpy(world, 1.0, amo_new, -c(m), rma, false);
            gaxpy(world, 1.0, bmo_new, c(m), vmb, false);
            gaxpy(world, 1.0, bmo_new, -c(m), rmb, false);
        }
        world.gop.fence();
        END_TIMER(world, "Subspace transform");
        if(param.maxsub <= 1){
            subspace.clear();
        } else  if(subspace.size() == param.maxsub){
            subspace.erase(subspace.begin());
            Q = Q(Slice(1, -1), Slice(1, -1));
        }

        vector<double> anorm = norm2(world, sub(world, amo, amo_new));
        vector<double> bnorm = norm2(world, sub(world, bmo, bmo_new));
        int nres = 0;
        for(unsigned int i = 0;i < amo.size();i++){
            if(anorm[i] > param.maxrotn){
                double s = param.maxrotn / anorm[i];
                nres++;
                if(world.rank() == 0){
                    if(nres == 1)
                        printf("  restricting step for alpha orbitals:");

                    printf(" %d", i);
                }
                amo_new[i].gaxpy(s, amo[i], 1.0 - s, false);
            }

        }

        if(nres > 0 && world.rank() == 0)
            printf("\n");

        nres = 0;
        for(unsigned int i = 0;i < bmo.size();i++){
            if(bnorm[i] > param.maxrotn){
                double s = param.maxrotn / bnorm[i];
                nres++;
                if(world.rank() == 0){
                    if(nres == 1)
                        printf("  restricting step for  beta orbitals:");

                    printf(" %d", i);
                }
                bmo_new[i].gaxpy(s, bmo[i], 1.0 - s, false);
            }

        }

        world.gop.fence();
        double rms, maxval;
        vector_stats(anorm, rms, maxval);
        if(world.rank() == 0)
            print("Norm of vector changes alpha: rms", rms, "   max", maxval);

        update_residual = maxval;
        if(bnorm.size()){
            vector_stats(bnorm, rms, maxval);
            if(world.rank() == 0)
                print("Norm of vector changes  beta: rms", rms, "   max", maxval);

            update_residual = max(update_residual, maxval);
        }
        START_TIMER(world);
        double trantol = vtol / min(30.0, double(amo.size()));
        normalize(world, amo_new);
        amo_new = transform(world, amo_new, Q3(matrix_inner(world, amo_new, amo_new)), trantol, true);
        truncate(world, amo_new);
        normalize(world, amo_new);
        if(param.nbeta && !param.spin_restricted){
            normalize(world, bmo_new);
            bmo_new = transform(world, bmo_new, Q3(matrix_inner(world, bmo_new, bmo_new)), trantol, true);
            truncate(world, bmo_new);
            normalize(world, bmo_new);
        }
        END_TIMER(world, "Orthonormalize");
        amo = amo_new;
        bmo = bmo_new;
    }

    void solve(World & world)
    {
        functionT arho_old, brho_old;
        functionT adelrhosq, bdelrhosq;
        const double dconv = max(FunctionDefaults<3>::get_thresh(), param.dconv);
        const double trantol = vtol / min(30.0, double(amo.size()));
        const double tolloc = 1e-3;
        double update_residual = 0.0, bsh_residual = 0.0;
        subspaceT subspace;
        tensorT Q;
        bool do_this_iter = true;
        // Shrink subspace until stop localizing/canonicalizing
        int maxsub_save = param.maxsub;
        param.maxsub = 2;
        
        for(int iter = 0;iter < param.maxiter;iter++){
            if(world.rank() == 0)
                printf("\nIteration %d at time %.1fs\n\n", iter, wall_time());
            
            if (iter > 0 && update_residual < 0.1) {
                do_this_iter = false;
                param.maxsub = maxsub_save;
            }

            if(param.localize && do_this_iter) {
                tensorT U = localize_PM(world, amo, aset, tolloc, 0.25, iter == 0);
                amo = transform(world, amo, U, trantol, true);
                truncate(world, amo);
                normalize(world, amo);
                if(!param.spin_restricted && param.nbeta){
                    U = localize_PM(world, bmo, bset, tolloc, 0.25, iter == 0);
                    bmo = transform(world, bmo, U, trantol, true);
                    truncate(world, bmo);
                    normalize(world, bmo);
                }
            }

            START_TIMER(world);
            functionT arho = make_density(world, aocc, amo);
            functionT brho;
            if(!param.spin_restricted && param.nbeta)
                brho = make_density(world, bocc, bmo);
            else
                brho = arho;

            END_TIMER(world, "Make densities");
            if(iter < 2 || (iter % 10) == 0){
                START_TIMER(world);
                loadbal(world, arho, brho, arho_old, brho_old, subspace);
                END_TIMER(world, "Load balancing");
            }
            double da = 0.0, db = 0.0;
            if(iter > 0){
                da = (arho - arho_old).norm2();
                db = (brho - brho_old).norm2();
                if(world.rank() == 0)
                    print("delta rho", da, db, "residuals", bsh_residual, update_residual);

            }
            arho_old = arho;
            brho_old = brho;
            functionT rho = arho + brho;
            rho.truncate();
            double enuclear = inner(rho, vnuc);
            START_TIMER(world);
            functionT vcoul = apply(*coulop, rho);
            END_TIMER(world, "Coulomb");
            double ecoulomb = 0.5 * inner(rho, vcoul);
            rho.clear(false);
            functionT vlocal = vcoul + vnuc;
            vcoul.clear(false);
            vlocal.truncate();
            double exca = 0.0, excb = 0.0;
            vecfuncT Vpsia = apply_potential(world, aocc, amo, arho, brho, adelrhosq, bdelrhosq, vlocal, exca);
            vecfuncT Vpsib;
            if(param.spin_restricted){
                if(!param.lda)  excb = exca;
            } 
            else if(param.nbeta) {
                Vpsib = apply_potential(world, bocc, bmo, brho, arho, bdelrhosq, adelrhosq, vlocal, excb);
            }

            double ekina = 0.0, ekinb = 0.0;
            tensorT focka = make_fock_matrix(world, amo, Vpsia, aocc, ekina);
            tensorT fockb = focka;

            if(param.spin_restricted)
                ekinb = ekina;
            else
                fockb = make_fock_matrix(world, bmo, Vpsib, bocc, ekinb);

            if (!param.localize) {
                if (do_this_iter) {
                    diag_fock_matrix(world, focka, amo, Vpsia, aeps, dconv);
                    if (!param.spin_restricted && param.nbeta) 
                        diag_fock_matrix(world, fockb, bmo, Vpsib, beps, dconv);
                }
            }
            
            if(world.rank() == 0){
                double enrep = molecule.nuclear_repulsion_energy();
                double ekinetic = ekina + ekinb;
                double exc = exca + excb;
                double etot = ekinetic + enuclear + ecoulomb + exc + enrep;
                printf("\n              kinetic %16.8f\n", ekinetic);
                printf("   nuclear attraction %16.8f\n", enuclear);
                printf("              coulomb %16.8f\n", ecoulomb);
                printf(" exchange-correlation %16.8f\n", exc);
                printf("    nuclear-repulsion %16.8f\n", enrep);
                printf("                total %16.8f\n\n", etot);
            }

            if(iter > 0){
                if(da < dconv * molecule.natom() && db < dconv * molecule.natom() && bsh_residual < 5.0*dconv){
                    if(world.rank() == 0) {
                        print("\nConverged!\n");
                    }

                    // Diagonalize to get the eigenvalues and if desired the final eigenvectors
                    tensorT U;
                    tensorT overlap = matrix_inner(world, amo, amo, true);
                    sygv(focka, overlap, 1, &U, &aeps);
                    if (!param.localize) {
                        amo = transform(world, amo, U, trantol, true);
                        truncate(world, amo);
                        normalize(world, amo);
                    }
                    if(param.nbeta && !param.spin_restricted){
                        overlap = matrix_inner(world, bmo, bmo, true);
                        sygv(fockb, overlap, 1, &U, &beps);
                        if (!param.localize) {
                            bmo = transform(world, bmo, U, trantol, true);
                            truncate(world, bmo);
                            normalize(world, bmo);
                        }
                    }

                    if(world.rank() == 0) {
                        print(" ");
                        print("alpha eigenvalues");
                        print(aeps);
                        if(param.nbeta && !param.spin_restricted){
                            print("beta eigenvalues");
                            print(beps);
                        }
                    }

                    if (param.localize) {
                        // Restore the diagonal elements for the analysis
                        for (unsigned int i=0; i<amo.size(); i++) aeps[i] = focka(i,i);
                        for (unsigned int i=0; i<bmo.size(); i++) beps[i] = fockb(i,i);
                    }

                    break;
                }

            }

            update_subspace(world, Vpsia, Vpsib, focka, fockb, subspace, Q, bsh_residual, update_residual);
        }

        if(world.rank() == 0) {
            if (param.localize) print("Orbitals are localized - energies are diagonal Fock matrix elements\n");
            else print("Orbitals are eigenvectors - energies are eigenvalues\n");
            print("Analysis of alpha MO vectors");
        }

        analyze_vectors(world, amo, aocc, aeps);
        if(param.nbeta && !param.spin_restricted){
            if(world.rank() == 0)
                print("Analysis of beta MO vectors");

            analyze_vectors(world, bmo, bocc, beps);
        }
        functionT arho = make_density(world, aocc, amo);
        functionT brho = arho;
        if (!param.spin_restricted) brho = make_density(world, bocc, bmo);

        arho.gaxpy(1.0, brho, 1.0);
        tensorT dv = derivatives(world, arho);
        if (world.rank() == 0) {
            print("\n Derivatives (a.u.)\n -----------\n");
            print("  atom        x            y            z          dE/dx        dE/dy        dE/dz");
            print(" ------ ------------ ------------ ------------ ------------ ------------ ------------");
            for (int i=0; i<molecule.natom(); i++) {
                const Atom& atom = molecule.get_atom(i);
                printf(" %5d %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n",
                       i, atom.x, atom.y, atom.z,
                       dv[i*3+0], dv[i*3+1], dv[i*3+2]);
            }
        }
    }
};

int main(int argc, char** argv) {
    initialize(argc, argv);

    World world(MPI::COMM_WORLD);

    try {
        // Load info for MADNESS numerical routines
        startup(world,argc,argv);
        FunctionDefaults<3>::set_pmap(pmapT(new LevelPmap(world)));

        cout.precision(6);

        // Process 0 reads input information and broadcasts
        Calculation calc(world, "input");

        // Warm and fuzzy for the user
        if (world.rank() == 0) {
            print("\n\n");
            print(" MADNESS Hartree-Fock and Density Functional Theory Program");
            print(" ----------------------------------------------------------\n");
            print("\n");
            calc.molecule.print();
            print("\n");
            calc.param.print(world);
        }

        // Come up with an initial OK data map
        if (world.size() > 1) {
            calc.set_protocol(world,1e-4);
            calc.make_nuclear_potential(world);
            calc.initial_load_bal(world);
        }

        // Make the nuclear potential, initial orbitals, etc.
        calc.set_protocol(world,1e-4);
        calc.make_nuclear_potential(world);
        calc.project_ao_basis(world);
        //calc.project(world);
        calc.initial_guess(world);
        calc.solve(world);

        calc.set_protocol(world,1e-6);
        calc.make_nuclear_potential(world);
        calc.project_ao_basis(world);
        calc.project(world);
        calc.solve(world);

//         calc.set_protocol(world,1e-8);
//         calc.make_nuclear_potential(world);
//         calc.project(world);
//         calc.solve(world);

    }
    catch (const MPI::Exception& e) {
        //        print(e);
        error("caught an MPI exception");
    }
    catch (const madness::MadnessException& e) {
        print(e);
        error("caught a MADNESS exception");
    }
    catch (const madness::TensorException& e) {
        print(e);
        error("caught a Tensor exception");
    }
    catch (const char* s) {
        print(s);
        error("caught a string exception");
    }
    catch (char* s) {
        print(s);
        error("caught a string exception");
    }
    catch (const std::string& s) {
        print(s);
        error("caught a string (class) exception");
    }
    catch (const std::exception& e) {
        print(e.what());
        error("caught an STL exception");
    }
    catch (...) {
        error("caught unhandled exception");
    }

    // Nearly all memory will be freed at this point
    world.gop.fence();
    world.gop.fence();
    ThreadPool::end();
    print_stats(world);

    finalize();

    return 0;
}

