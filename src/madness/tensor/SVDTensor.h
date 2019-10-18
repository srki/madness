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
#ifndef SRC_MADNESS_TENSOR_SVDTENSOR_H_
#define SRC_MADNESS_TENSOR_SVDTENSOR_H_

#include <madness/tensor/srconf.h>

namespace madness{


template<typename T>
class SVDTensor : public SRConf<T> {
public:
	using SRConf<T>::SRConf;

	SVDTensor() : SRConf<T>() {}

	SVDTensor(const Tensor<T>& rhs, const double eps) {
		if (rhs.has_data()) {
//			*this=compute_svd(rhs,eps);
			*this=compute_randomized_svd(rhs,eps);
		}
	}

	SVDTensor(const SVDTensor<T>& rhs) = default;

	SVDTensor(const SRConf<T>& rhs) : SRConf<T>(rhs) {    }

	SVDTensor(const std::vector<long>& dims) : SRConf<T>(dims.size(),dims.data(),dims.size()/2) {    }

	SVDTensor(const long ndims, const long* dims) : SRConf<T>(ndims,dims,ndims/2) {    }

	SVDTensor(const Tensor<double>& weights, const Tensor<T>& vector1,
			const Tensor<T>& vector2, const long& ndim,
			const long* dims) : SRConf<T>(weights, vector1, vector2, ndim, dims, ndim/2) {}

	SVDTensor& operator=(const T& number) {
		SRConf<T>& base=*this;
		base=number;
		return *this;
	}

	long size() const {
		return SRConf<T>::nCoeff();
	}

	/// reduce the rank using SVD
	SVDTensor compute_svd(const Tensor<T>& tensor,
			const double& eps, std::array<long,2> vectordim={0,0}) const;

	/// reduce the rank using SVD
	SVDTensor compute_randomized_svd(const Tensor<T>& tensor,
			const double& eps, std::array<long,2> vectordim={0,0}) const;

	/// compute the range of the matrix
	static Tensor<T> compute_range(const Tensor<T>& matrix,
			const double& eps, std::array<long,2> vectordim={0,0});

	static typename Tensor<T>::scalar_type check_range(const Tensor<T>& matrix,
			const Tensor<T>& range) {
		Tensor<T> residual=matrix-inner(range,inner(conj(range),matrix,0,0));
		return residual.normf();
	}

	/// following Alg. 5.1 of HMT 2011
	static SVDTensor<T> compute_svd_from_range(const Tensor<T>& range,
			const Tensor<T>& matrix);

	static Tensor<T> make_SVD_decaying_matrix(const Tensor<T>& matrix, const int n=1) {
		Tensor<T> U,VT;
		Tensor<typename Tensor<T>::scalar_type> s;
		svd(matrix,U,s,VT);
		for (long i=0; i<s.size(); ++i) {
			s(i)*=exp(-i/n);		// make singular values decay exponentially
			U(_,i)*=s(i);
		}
		return inner(U,VT);
	}

	SVDTensor<T>& emul(const SVDTensor<T>& other) {
		const SRConf<T>& base=other;
		SRConf<T>::emul(base);
		return *this;
	}

	SVDTensor<T>& gaxpy(T alpha, const SVDTensor<T>& rhs, T beta) {
		this->scale(alpha);
		this->append(rhs,beta);
		return *this;
	}

private:

	/// resize Tensor to a matrix
	static Tensor<T> resize_to_matrix(const Tensor<T>& matrix, std::array<long,2> vectordim={0,0}) {
		// SVD works only with matrices (2D)
		if (vectordim[0]==0) {
			long dim1=matrix.ndim()/2;	// integer division
			vectordim[0]=vectordim[1]=1;
			for (long i=0; i<dim1; ++i) vectordim[0]*=matrix.dim(i);
			for (long i=dim1; i<matrix.ndim(); ++i) vectordim[1]*=matrix.dim(i);
			MADNESS_ASSERT(vectordim[0]*vectordim[1]==matrix.size());
		}
		Tensor<T> values_eff=matrix.reshape(vectordim[0],vectordim[1]);
		MADNESS_ASSERT(values_eff.ndim()==2);
		return values_eff;
	}

public:

	template <typename R, typename Q>
	friend SVDTensor<TENSOR_RESULT_TYPE(R,Q)> transform(
			const SVDTensor<R>& t, const Tensor<Q>& c);

	template <typename R, typename Q>
	friend SVDTensor<TENSOR_RESULT_TYPE(R,Q)> general_transform(
			const SVDTensor<R>& t, const Tensor<Q> c[]);

	template <typename R, typename Q>
	friend SVDTensor<TENSOR_RESULT_TYPE(R,Q)> transform_dir(
			const SVDTensor<R>& t, const Tensor<Q>& c, const int axis);

    template <typename R, typename Q>
    friend SVDTensor<TENSOR_RESULT_TYPE(R,Q)> outer(
    		const SVDTensor<R>& t1, const SVDTensor<Q>& t2);

    friend SVDTensor<T> copy(const SVDTensor<T>& rhs) {
    	const SRConf<T>& base=rhs;
    	return SVDTensor<T>(copy(base));
    }

};

template <typename R, typename Q>
SVDTensor<TENSOR_RESULT_TYPE(R,Q)> transform(
		const SVDTensor<R>& t, const Tensor<Q>& c) {
	return SVDTensor<TENSOR_RESULT_TYPE(R,Q)>(t.transform(c));
}

template <typename R, typename Q>
SVDTensor<TENSOR_RESULT_TYPE(R,Q)> general_transform(
		const SVDTensor<R>& t, const Tensor<Q> c[]) {
	return SVDTensor<TENSOR_RESULT_TYPE(R,Q)>(t.general_transform(c));
}

template <typename R, typename Q>
SVDTensor<TENSOR_RESULT_TYPE(R,Q)> transform_dir(
		const SVDTensor<R>& t, const Tensor<Q>& c, const int axis) {
	return SVDTensor<TENSOR_RESULT_TYPE(R,Q)>(t.transform_dir(c, axis));
}

}



#endif /* SRC_MADNESS_TENSOR_SVDTENSOR_H_ */
