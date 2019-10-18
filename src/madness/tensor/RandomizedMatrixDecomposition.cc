
#include <madness/tensor/RandomizedMatrixDecomposition.h>
#include <madness/constants.h>
#include <madness/tensor/tensor_lapack.h>


namespace madness {


/// compute the range of the matrix, following Alg 4.2 of Halko, Martinsson, Tropp, 2011
template<typename T>
Tensor<T> RandomizedMatrixDecomposition<T>::compute_range(const Tensor<T>& tensor,
		const double& eps, std::array<long,2> vectordim) {

	typedef typename Tensor<T>::scalar_type scalar_type;

	// fast return if possible
	Tensor<T> Q(0l,0l);
	if (tensor.normf()<eps) return Q;

	const Tensor<T> matrix=resize_to_matrix(tensor,vectordim);
	const long m=matrix.dim(0);
	const long n=matrix.dim(1);

	maxrank=std::min(maxrank,matrix.dim(0));

	// random trial vectors (transpose for efficiency)
	Tensor<T> omegaT(oversampling,n);
	omegaT.fillrandom();

	// compute guess for the range
	Tensor<T> Y=inner(matrix,omegaT,1,1);

	std::vector<scalar_type> columnnorm(Y.dim(0));
	for (long i=0; i<Y.dim(0); ++i) columnnorm[i]=Y(i,_).normf();
	scalar_type maxnorm=*std::max_element(columnnorm.begin(),columnnorm.end());

	bool not_complete=(maxnorm>(eps/10*sqrt(2* madness::constants::pi)));

	Tensor<T> R;	// R is not needed anymore..
	if (not_complete) {
		qr(Y,R);
		Q=Y;
	} else {
		range=Q;
		return range;
	}

	while (not_complete) {
		omegaT.fillrandom();
		Y=inner(matrix,omegaT,1,1);
		Tensor<T> Y0=Y-inner(Q,inner(conj(Q),Y,0,0));

		// check residual norm, exit if converged
		std::vector<scalar_type> columnnorm0(Y0.dim(0));
		for (long i=0; i<Y0.dim(0); ++i) columnnorm0[i]=Y0(i,_).normf();
		maxnorm=*std::max_element(columnnorm0.begin(),columnnorm0.end());
		if (maxnorm<(eps/10*sqrt(2*constants::pi))) break;


		// concatenate the ranges
		Tensor<T> Q0(m,Q.dim(1)+Y.dim(1));
		Q0(_,Slice(0,Q.dim(1)-1))=Q;
		Q0(_,Slice(Q.dim(1),-1))=Y0;

		// orthonormalize the ranges
		qr(Q0,R);
		Q=Q0;

		if (Q.dim(1)>maxrank) break;
		if (Q.dim(1)==matrix.dim(0)) break;
		if (Q.dim(1)>matrix.dim(0)) MADNESS_EXCEPTION("faulty QR step in randomized range finder",1);;

	}
	range=Q;
	return range;
}



// explicit instantiation
template class RandomizedMatrixDecomposition<float>;
template class RandomizedMatrixDecomposition<double>;
template class RandomizedMatrixDecomposition<float_complex>;
template class RandomizedMatrixDecomposition<double_complex>;


} /* namespace madness */
