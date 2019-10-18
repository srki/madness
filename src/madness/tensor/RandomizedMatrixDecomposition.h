

#ifndef SRC_MADNESS_TENSOR_RANDOMIZEDMATRIXDECOMPOSITION_H_
#define SRC_MADNESS_TENSOR_RANDOMIZEDMATRIXDECOMPOSITION_H_

#include <madness/tensor/tensor.h>
using namespace madness;

namespace madness {


/// simple factory pattern for the RandomizedMatrixDecomposition
struct RMDFactory {
	long maxrank_=LONG_MAX;
	long oversampling_=10;

	RMDFactory& maxrank(const long mr) {
		maxrank_=mr;
		return *this;
	}
	RMDFactory& oversampling(const long os) {
		oversampling_=os;
		return *this;
	}

};


template<typename T>
class RandomizedMatrixDecomposition {

public:
	typedef typename TensorTypeData<T>::scalar_type scalar_type;

	RandomizedMatrixDecomposition() = default;

	RandomizedMatrixDecomposition(const RandomizedMatrixDecomposition& other) = default;

	RandomizedMatrixDecomposition(const RMDFactory& factory)
		: maxrank(factory.maxrank_), oversampling(factory.oversampling_) {
	}

	RandomizedMatrixDecomposition(const Tensor<T>& matrix, const double thresh) {
		range=compute_range(matrix,thresh,{0,0});
	}

	/// check if the computed rank is too large
	bool exceeds_maxrank() const {return (range.dim(1)>maxrank);}

	Tensor<T> get_range() const {return range;}

	/// compute the range of the matrix

	/// follows Halko, Martinsson, Tropp (2011), Alg. 4.2
	/// the final range will satisfy for the input tensor A: || A - Q Q^T A || < eps
	/// method will change member variables "range" and "maxrank"
	/// @param[in]		tensor		the input tensor/matrix A (see matrixdim if not a matrix)
	/// @param[in]		eps			the accuracy of the representation
	/// @param[in]		matrixdim	how to reshape the tensor if it is not a matrix
	/// @return 		range		the range of the input tensor
	Tensor<T> compute_range(const Tensor<T>& tensor,
			const double& eps, std::array<long,2> matrixdim={0,0});

	/// return the error of the range Q

	/// explicitly compute the error measure || A - Q Q^T A ||
	static scalar_type check_range(const Tensor<T>& matrix,
			const Tensor<T>& range) {
		Tensor<T> residual=matrix-inner(range,inner(conj(range),matrix,0,0));
		return residual.normf();
	}

	/// helper function
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

	void set_maxrank(const long max_rank) {
		maxrank=max_rank;
	}

private:

	/// oversampling parameter
	long oversampling=10;

	/// maximum rank to abort the decomposition
	long maxrank=LONG_MAX;

	/// the range that spans the input matrix
	Tensor<T> range=Tensor<T>(0l,0l);

};

} /* namespace madness */

#endif /* SRC_MADNESS_TENSOR_RANDOMIZEDMATRIXDECOMPOSITION_H_ */
