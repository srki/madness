/*
  This file is part of MADNESS.

  Copyright (C) 2007,2010 Oak Ridge National Laboratory

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
//#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include <madness/mra/mra.h>
#include <madness/mra/operator.h>
#include <madness/constants.h>

/*!
  \file heat2.cc
  \brief Example Green function for the 3D heat equation with a linear term
  \defgroup heatex2 Evolve in time 3D heat equation with a linear term
  \ingroup examples

  The source is <a href=http://code.google.com/p/m-a-d-n-e-s-s/source/browse/local/trunk/src/apps/examples/heat2.cc>here</a>.

  \par Points of interest
  - application of a function of a function to exponentiate the potential
  - use of a functor to compute the solution at an arbitrary future time
  - convolution with the Green's function


  \par Background

  This adds to the complexity of the other \ref exampleheat "heat equation example"
  by including a linear term.  Specifically, we solve
  \f[
  \frac{\partial u(x,t)}{\partial t} = c \nabla^2 u(x,t) + V_p(x,t) u(x,t)
  \f]
  If \f$ V_p = 0 \f$ time evolution operator is
  \f[
  G_0(x,t) = \frac{1}{\sqrt{4 \pi c t}} \exp \frac{-x^2}{4 c t}
  \f]
  For non-zero \f$ V_p \f$ the time evolution is performed using the Trotter splitting
  \f[
  G(x,t) = G_0(x,t/2) * \exp(V_p t) * G_0(x,t/2) + O(t^3)
  \f]
  In order to form an exact solution for testing, we choose \f$ V_p(x,t)=\mbox{constant} \f$
  but the solution method is not limited to this choice.

*/

using namespace madness;

static const double L		= 20;     // Half box size
static const long	k		= 8;        // wavelet order
static const double thresh	= 1e-10; // precision   // w/o diff. and 1e-12 -> 64 x 64
static const double c		= 2.0;       //
static const double tstep	= 0.1;
static const double alpha	= 1.9; // Exponent
static const double VVV		= 0.2;  // Vp constant value

#define PI 3.1415926535897932385
#define LO 0.0000000000
#define HI 4.0000000000

static double sin_amp		= 1.0;
static double cos_amp		= 1.0;
static double sin_freq		= 1.0;
static double cos_freq		= 1.0;
static double sigma_x		= 1.0;
static double sigma_y		= 1.0;
static double sigma_z		= 1.0;
static double center_x		= 0.0;
static double center_y		= 0.0;
static double center_z		= 0.0;
static double gaussian_amp	= 1.0;
static double sigma_sq_x	= sigma_x*sigma_x;
static double sigma_sq_y	= sigma_y*sigma_y;
static double sigma_sq_z	= sigma_z*sigma_z;

#define FUNC_SIZE	50

// region functions
template<size_t NDIM>
using coord = madness::Vector<double, NDIM>;

inline double gauss1D(const coord<1> &x, const coord<1> &a, const coord<1> &b, const coord<1> &c) {
    return a[0] * exp(-pow(x[0] - b[0], 2) / (2 * c[0] * c[0]));
}

inline double gauss2D(const coord<2> &x, const coord<2> &a, const coord<2> &b, const coord<2> &c) {
    return gauss1D(coord<1>{x[0]}, coord<1>{a[0]}, coord<1>{b[0]}, coord<1>{c[0]}) *
           gauss1D(coord<1>{x[1]}, coord<1>{a[1]}, coord<1>{b[1]}, coord<1>{c[1]});
}

inline double gauss3D(const coord<3> &x, const coord<3> &a, const coord<3> &b, const coord<3> &c) {
    return gauss1D(coord<1>{x[0]}, coord<1>{a[0]}, coord<1>{b[0]}, coord<1>{c[0]}) *
           gauss1D(coord<1>{x[1]}, coord<1>{a[1]}, coord<1>{b[1]}, coord<1>{c[1]}) *
           gauss1D(coord<1>{x[2]}, coord<1>{a[2]}, coord<1>{b[2]}, coord<1>{c[2]});
}

/**
 * Function analytic form.
 *  exp(-(x-0.5)^2 / (2 * (1/15)^2)) * exp(-(y-0.5)^2 / (2 * (1/15)^2)) * exp(-(z-0.5)^2 / (2 * (1/15)^2))
 * @param x
 * @return
 */
inline double test3DFn0(const coord<3> &x) {
    return gauss3D(x, coord<3>{1, 1, 1}, coord<3>{0.5, 0.5, 0.5}, coord<3>{1.0 / 15, 1.0 / 15, 1.0 / 15});
}

/**
 * Function analytic form.
 *  exp(-(x-0.4)^2 / (2 * (1/15)^2)) * exp(-(y-0.5)^2 / (2 * (1/15)^2)) * exp(-(z-0.5)^2 / (2 * (1/15)^2))
 * @param x
 * @return
 */
inline double test3DFn1(const coord<3> &x) {
    return gauss3D(x, coord<3>{1, 1, 1}, coord<3>{0.4, 0.5, 0.5}, coord<3>{1.0 / 15, 1.0 / 15, 1.0 / 15});
}

/**
 * Function analytic form.
 *  exp(-(x-0.5)^2 / (2 * (1/15)^2)) * exp(-(y-0.4)^2 / (2 * (1/15)^2)) * exp(-(z-0.5)^2 / (2 * (1/15)^2))
 * @param x
 * @return
 */
inline double test3DFn2(const coord<3> &x) {
    return gauss3D(x, coord<3>{1, 1, 1}, coord<3>{0.5, 0.4, 0.5}, coord<3>{1.0 / 15, 1.0 / 15, 1.0 / 15});
}

/**
 * Function analytic form.
 *  exp(-(x-0.5)^2 / (2 * (1/15)^2)) * exp(-(y-0.5)^2 / (2 * (1/15)^2)) * exp(-(z-0.4)^2 / (2 * (1/15)^2))
 * @param x
 * @return
 */
inline double test3DFn3(const coord<3> &x) {
    return gauss3D(x, coord<3>{1, 1, 1}, coord<3>{0.5, 0.5, 0.4}, coord<3>{1.0 / 15, 1.0 / 15, 1.0 / 15});
}

// endregion

double rtclock();


int main(int argc, char** argv)
{
    initialize(argc, argv);
    World world(SafeMPI::COMM_WORLD);

    startup(world, argc, argv);

    FunctionDefaults<3>::set_k(k);
    FunctionDefaults<3>::set_thresh(thresh);
    FunctionDefaults<3>::set_refine(true);
    FunctionDefaults<3>::set_autorefine(false);
    FunctionDefaults<3>::set_initial_level(0);
//    FunctionDefaults<3>::set_cubic_cell(-L, L);


//    FunctionDefaults<3>::set_max_refine_level(14);

    std::vector<decltype(test3DFn0)*> fns;
    for (size_t i = 0; i < FUNC_SIZE; i++) {
        fns.emplace_back(test3DFn0);
    }

    if (world.rank() == 0) print ("====================================================");
    if (world.rank() == 0) printf("   Initializing Functions\n");
    if (world.rank() == 0) printf("     %d Functions\n", FUNC_SIZE);
    if (world.rank() == 0) print ("====================================================");
    world.gop.fence();

    // 2 * N Functions
    real_function_3d  h[FUNC_SIZE];

    real_function_3d  temp_factory_h[FUNC_SIZE];
    real_function_3d* temp_h[FUNC_SIZE];


    int i, j;
    double clkbegin, clkend;
    clkbegin = rtclock();

    for (i=0; i<FUNC_SIZE; i++)
    {
//        h[i]				= real_factory_3d(world).f(fns[i]);
        h[i]				= real_factory_3d(world).f(test3DFn0);
        temp_factory_h[i]	= real_factory_3d(world);
        temp_h[i]			= new real_function_3d(temp_factory_h[i]);
    }

    for (i = 0; i < FUNC_SIZE; i++) {
        h[i].truncate();
    }

    clkend = rtclock() - clkbegin;
    if (world.rank() == 0) printf("Running Time: %f\n", clkend);

    if (world.rank() == 0) print ("====================================================");
    if (world.rank() == 0) print ("==      MADNESS       ==============================");
    if (world.rank() == 0) print ("====================================================");
    world.gop.fence();

    clkbegin = rtclock();
    double resultInner[FUNC_SIZE][FUNC_SIZE] = {};

    for (i=0; i<FUNC_SIZE; i++)
        h[i].compress();


//    for (i=0; i<FUNC_SIZE; i++)
//        for (j=i; j<FUNC_SIZE; j++)
//            resultInner[i][j] = h[i].inner(h[j]);

    world.gop.barrier();
    clkend = rtclock() - clkbegin;
    if (world.rank() == 0) printf("Running Time: %f\n", clkend);
    world.gop.barrier();

    for (i=0; i<FUNC_SIZE; i++)
        for (j=0; j<FUNC_SIZE; j++)
        {
            //if (world.rank() == 0) printf ("%d:%d = %.15f\n", i, j, resultInner[i][j]);
        }

    world.gop.fence();

    finalize();
    return 0;
}

double rtclock()
{struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0)
        printf("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

