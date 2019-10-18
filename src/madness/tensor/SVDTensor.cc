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

  $Id: srconf.h 2816 2012-03-23 14:59:52Z 3ru6ruWu $
 */



#include <madness/tensor/SVDTensor.h>
#include <madness/constants.h>

using namespace madness;

namespace madness {

/// reduce the rank using SVD
template<typename T>
SVDTensor<T> SVDTensor<T>::compute_svd(const Tensor<T>& values,
		const double& eps, std::array<long,2> vectordim) const {

	SVDTensor<T> result(values.ndim(),values.dims());

	// fast return if possible
	if (values.normf()<eps) return result;

	Tensor<T> values_eff=resize_to_matrix(values,vectordim);

	// output from svd
	Tensor<T> U;
	Tensor<T> VT;
	Tensor< typename Tensor<T>::scalar_type > s;

	svd(values_eff,U,s,VT);

	// find the maximal singular value that's supposed to contribute
	// singular values are ordered (largest first)
	const double thresh=eps;
	long i=SRConf<T>::max_sigma(thresh,s.dim(0),s);

	// convert SVD output to our convention
	if (i>=0) {
		// copy to have contiguous and tailored singular vectors
		result.set_vectors_and_weights(copy(s(Slice(0,i))), copy(transpose(U(_,Slice(0,i)))),
				copy(VT(Slice(0,i),_)));
	}
	return result;
}


/// reduce the rank using SVD
template<typename T>
SVDTensor<T> SVDTensor<T>::compute_randomized_svd(const Tensor<T>& tensor,
		const double& eps, std::array<long,2> vectordim) const {

	Tensor<T> Q=SVDTensor<T>::compute_range(tensor,eps*0.1,vectordim);
	if (Q.size()==0) return SVDTensor<T>(tensor.ndim(),tensor.dims());
	SVDTensor<T> result=SVDTensor<T>::compute_svd_from_range(Q,tensor);
	// TODO: fixme
//	result.orthonormalize(eps);
	return result;

}

/// compute the range of the matrix, following Alg 4.2 of Halko, Martinsson, Tropp, 2011
template<typename T>
Tensor<T> SVDTensor<T>::compute_range(const Tensor<T>& tensor,
		const double& eps, std::array<long,2> vectordim) {

	typedef typename Tensor<T>::scalar_type scalar_type;

	// fast return if possible
	Tensor<T> result;
	if (tensor.normf()<eps) return result;

	// random parameters
	const long oversampling=10;

	const Tensor<T> matrix=resize_to_matrix(tensor,vectordim);
	const long m=matrix.dim(0);
	const long n=matrix.dim(1);

	// random trial vectors (transpose for efficiency)
	Tensor<T> omegaT(oversampling,n);
	omegaT.fillrandom();

	// compute guess for the range
	Tensor<T> Y=inner(matrix,omegaT,1,1);

	std::vector<scalar_type> columnnorm(Y.dim(0));
	for (long i=0; i<Y.dim(0); ++i) columnnorm[i]=Y(i,_).normf();
	scalar_type maxnorm=*std::max_element(columnnorm.begin(),columnnorm.end());

	bool not_complete=(maxnorm>(eps/10*sqrt(2* madness::constants::pi)));

	Tensor<T> Q,R;	// R is not needed anymore..
	if (not_complete) {
		qr(Y,R);
		Q=Y;
	} else {
		return result;
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

		if (Q.dim(1)==matrix.dim(0)) break;
		if (Q.dim(1)>matrix.dim(0)) MADNESS_EXCEPTION("faulty QR step in randomized range finder",1);;

	}
	return Q;
}

template<typename T>
SVDTensor<T> SVDTensor<T>::compute_svd_from_range(const Tensor<T>& Q,
		const Tensor<T>& tensor) {
	typedef typename Tensor<T>::scalar_type scalar_type;

	MADNESS_ASSERT(tensor.size()%Q.dim(0) == 0);
	std::array<long,2> vectordim={Q.dim(0),tensor.size()/Q.dim(0)};
	const Tensor<T> matrix=resize_to_matrix(tensor,vectordim);

	Tensor<T> B=inner(conj(Q),matrix,0,0);
	Tensor<T> U,VT;
	Tensor<scalar_type> s;
	svd(B,U,s,VT);
	U=copy(conj_transpose(inner(Q,U)));
	SVDTensor<T> result(tensor.ndim(),tensor.dims());
	result.set_vectors_and_weights(s,U,VT);
	return result;
}


// explicit instantiation
template class SVDTensor<float>;
template class SVDTensor<double>;
template class SVDTensor<float_complex>;
template class SVDTensor<double_complex>;

} // namespace madness

