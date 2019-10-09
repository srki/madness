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

#ifndef MADNESS_TENSOR_LOWRANKTENSOR_H_
#define MADNESS_TENSOR_LOWRANKTENSOR_H_

#include <memory>
#include <vector>
#include <variant>

#include <madness/world/madness_exception.h>
#include <madness/world/print.h>
#include <madness/tensor/gentensor.h>
#include <madness/tensor/slice.h>
#include "srconf.h"
#include <madness/tensor/SVDTensor.h>
#include "tensor.h"
#include "tensortrain.h"
#include "type_data.h"

namespace madness {


// forward declaration
template <class T> class SliceLowRankTensor;


template<typename T>
class LowRankTensor {

public:

    /// C++ typename of the real type associated with a complex type.
    typedef typename TensorTypeData<T>::scalar_type scalar_type;

    /// C++ typename of the floating point type associated with scalar real type
    typedef typename TensorTypeData<T>::float_scalar_type float_scalar_type;

    /// empty ctor
    LowRankTensor() = default;

    /// copy ctor, shallow
    LowRankTensor(const LowRankTensor<T>& other) = default;

    /// ctor with dimensions; constructs tensor filled with zeros
    LowRankTensor(const std::vector<long>& dim, const TensorType& tt) {
        if (tt==TT_FULL) tensor=Tensor<T>(dim);
        if (tt==TT_2D) tensor=SVDTensor<T>(dim);
        if (tt==TT_TENSORTRAIN) tensor=TensorTrain<T>(dim);
    }

    /// ctor with dimensions; constructs tensor filled with zeros
    LowRankTensor(const std::vector<long>& dim, const TensorArgs& targs) :
        LowRankTensor(dim, targs.tt) {
    }

    /// ctor with dimensions; all dims have the same value k
    LowRankTensor(const TensorType& tt, const long k, const long ndim) :
        LowRankTensor(std::vector<long>(ndim,k), tt) {
    }

    /// ctor with a regular Tensor and arguments, deep
    LowRankTensor(const Tensor<T>& rhs, const double& thresh, const TensorType& tt) :
        LowRankTensor(rhs,TensorArgs(thresh,tt)) {
    }

    /// ctor with a regular Tensor and arguments, deep
    LowRankTensor(const Tensor<T>& rhs, const TensorArgs& targs) {
        if (targs.tt==TT_FULL) tensor=copy(Tensor<T>(rhs));
        else if (targs.tt==TT_2D) tensor=SVDTensor<T>(rhs,targs.thresh*facReduce());
        else if (targs.tt==TT_TENSORTRAIN) tensor=TensorTrain<T>(rhs,targs.thresh);
        else {
        	MADNESS_EXCEPTION("unknown tensor type in LowRankTensor constructor",1);
        }
    }

    /// ctor with a regular Tensor, deep
    LowRankTensor(const Tensor<T>& other) : tensor(other) {}

    /// ctor with a TensorTrain as argument, shallow
    LowRankTensor(const TensorTrain<T>& other) : tensor(other) {}

    /// ctor with a SVDTensor as argument, shallow
    LowRankTensor(const SVDTensor<T>& other) : tensor(other) {}

    /// ctor with a SliceLowRankTensor as argument, deep
    LowRankTensor(const SliceLowRankTensor<T>& other) {
        *this=other;
    }

    /// shallow assignment operator
    LowRankTensor& operator=(const LowRankTensor<T>& other) {
        if (this!=&other) tensor=other.tensor;
        return *this;
    }

    /// shallow assignment operator
    LowRankTensor& operator=(const Tensor<T>& other) {
        tensor=other;
        return *this;
    }

    /// shallow assignment operator
    LowRankTensor& operator=(const SVDTensor<T>& other) {
        tensor=other;
        return *this;
    }

    /// shallow assignment operator
    LowRankTensor& operator=(const TensorTrain<T>& other) {
        tensor=other;
        return *this;
    }

    /// deep assignment with slices: g0 = g1(s)
    LowRankTensor& operator=(const SliceLowRankTensor<T>& other) {
        const std::vector<Slice>& s=other.thisslice;
        if (this->tensor_type()==TT_FULL)
            tensor=copy(other.get_tensor()(s));
        else if (this->tensor_type()==TT_2D)
            tensor=other.get_svdtensor().copy_slice(s);
        else if (this->tensor_type()==TT_TENSORTRAIN)
            tensor=copy(other.get_tensortrain(),s);
        else {
            MADNESS_EXCEPTION("you should not be here",1);
        }
        return *this;
    }

    /// Type conversion makes a deep copy
    template <class Q> operator LowRankTensor<Q>() const { // type conv => deep copy

        LowRankTensor<Q> result;
        if (this->tensor_type()==TT_FULL) {
            result=Tensor<Q>(get_tensor());
        } else if (this->tensor_type()==TT_2D) {
            MADNESS_EXCEPTION("no type conversion for TT_2D yes=t",1);
        } else if (this->tensor_type()==TT_TENSORTRAIN) {
            MADNESS_EXCEPTION("no type conversion for TT_2D yes=t",1);
        } else {
            MADNESS_EXCEPTION("you should not be here",1);
        }
        return result;
    }

    SVDTensor<T>& get_svdtensor() {
         if (SVDTensor<T>* test=std::get_if<SVDTensor<T> >(&tensor)) return *test;
         MADNESS_EXCEPTION("failure to return SVDTensor from LowRankTensor",1);
     }

    const SVDTensor<T>& get_svdtensor() const {
         if (const SVDTensor<T>* test=std::get_if<SVDTensor<T> >(&tensor)) return *test;
         MADNESS_EXCEPTION("failure to return SVDTensor from LowRankTensor",1);
    }

    Tensor<T>& get_tensor() {
        if (Tensor<T>* test=std::get_if<Tensor<T> >(&tensor)) return *test;
        MADNESS_EXCEPTION("failure to return Tensor from LowRankTensor",1);
     }

    const Tensor<T>& get_tensor() const {
        if (const Tensor<T>* test=std::get_if<Tensor<T> >(&tensor)) return *test;
        MADNESS_EXCEPTION("failure to return Tensor from LowRankTensor",1);
     }

    TensorTrain<T>& get_tensortrain() {
        if (TensorTrain<T>* test=std::get_if<TensorTrain<T> >(&tensor)) return *test;
        MADNESS_EXCEPTION("failure to return TensorTrain from LowRankTensor",1);
    }

    const TensorTrain<T>& get_tensortrain() const {
        if (const TensorTrain<T>* test=std::get_if<TensorTrain<T> >(&tensor)) return *test;
        MADNESS_EXCEPTION("failure to return TensorTrain from LowRankTensor",1);
    }

    /// general slicing, shallow; for temporary use only!
    SliceLowRankTensor<T> operator()(const std::vector<Slice>& s) {
        return SliceLowRankTensor<T>(*this,s);
    }

    /// general slicing, shallow; for temporary use only!
    const SliceLowRankTensor<T> operator()(const std::vector<Slice>& s) const {
        return SliceLowRankTensor<T>(*this,s);
    }


//    LowRankTensor& operator=(const double& fac) {
//        MADNESS_EXCEPTION("no assignment of numbers in LowRankTensor",1);
//        return *this;
//    }

    /// deep copy
    friend LowRankTensor copy(const LowRankTensor& other) {
    	LowRankTensor<T> result;
        std::visit([&result](auto& obj) {result=copy(obj);}, other.tensor);
        return result;
    }

    /// return the tensor type
    TensorType tensor_type() const {
    	if (std::holds_alternative<SVDTensor<T> >(tensor)) return TT_2D;
    	if (std::holds_alternative<Tensor<T> >(tensor)) return TT_FULL;
    	if (std::holds_alternative<TensorTrain<T> >(tensor)) return TT_TENSORTRAIN;
    	MADNESS_EXCEPTION("confused tensor types ",1);
    }

    template<typename Q, typename R>
    friend bool is_same_tensor_type(const LowRankTensor<R>& rhs, const LowRankTensor<Q>& lhs) {
    	return (rhs.tensor.index()==lhs.tensor.index());
    }

    /// convert this to a new LowRankTensor of given tensor type
    LowRankTensor convert(const TensorArgs& targs) const {

        // fast return if old and new type are identical
    	if (targs.tt==tensor_type()) return copy(*this);

    	// target is full tensor
    	if (targs.tt==TT_FULL) return this->full_tensor_copy();

    	// source is full tensor: construct the corresponding representation
    	if (tensor_type()==TT_FULL) return LowRankTensor<T>(get_tensor(),targs);

    	// TT_TENSORTRAIN TO TT_2D
    	if ((tensor_type()==TT_TENSORTRAIN) and (targs.tt==TT_2D)) {
			Tensor<T> U,VT;
			Tensor< typename Tensor<T>::scalar_type > s;
			get_tensortrain().two_mode_representation(U, VT, s);
			long rank=s.size();
			if (rank==0) return SVDTensor<T>(get_tensortrain().ndim(),get_tensortrain().dims(),ndim()/2);

			long n=1,m=1;
			for (int i=0; i<U.ndim()-1; ++i) n*=U.dim(i);
			for (int i=1; i<VT.ndim(); ++i) m*=VT.dim(i);
			MADNESS_ASSERT(rank*n==U.size());
			MADNESS_ASSERT(rank*m==VT.size());
			U=copy(transpose(U.reshape(n,rank)));   // make it contiguous
			VT=VT.reshape(rank,m);
			SVDTensor<T> svdtensor(s, U, VT, ndim(), dim(0));
    		return LowRankTensor<T>(svdtensor);
    	}

    	print("conversion from type ",tensor_type(), "to type", targs.tt,"not supported");
    	MADNESS_EXCEPTION("type conversion not supported in LowRankTensor::convert ",1);
    	return *this;
    }

    long ndim() const {return ptr()->ndim();}

    /// return the number of entries in dimension i
    long dim(const int i) const {return ptr()->dim(i);}

    void normalize() {
        if (tensor_type()==TT_2D) get_svdtensor().normalize();
    }

    float_scalar_type normf() const {
    	float_scalar_type norm;
        std::visit([&norm](auto& obj) {norm=obj.normf();}, tensor);
        return norm;
    }

    float_scalar_type svd_normf() const {
        MADNESS_EXCEPTION("recode svd_normf in lowranktensor",1);
        return 0.0;
    }


    /// Inplace multiplication by scalar of supported type (legacy name)

    /// @param[in] x Scalar value
    /// @return %Reference to this tensor
    template <typename Q>
    typename IsSupported<TensorTypeData<Q>,LowRankTensor<T>&>::type
    scale(Q fac) {
        std::visit([&fac](auto& obj) {obj.scale(T(fac));}, tensor);
        return *this;
    }

    Tensor<T> full_tensor_copy() const {
        if (tensor_type()==TT_NONE) return Tensor<T>();
        else if (tensor_type()==TT_FULL) return copy(get_tensor());
        else if (tensor_type()==TT_2D) return get_svdtensor().reconstruct();
        else if (tensor_type()==TT_TENSORTRAIN) return get_tensortrain().reconstruct();
        else {
            MADNESS_EXCEPTION("you should not be here",1);
        }
        return Tensor<T>();
    }

    const Tensor<T>& full_tensor() const {
    	return get_tensor();
    }

    Tensor<T>& full_tensor() {
    	return get_tensor();
    }


    /// reconstruct this to return a full tensor
    Tensor<T> reconstruct_tensor() const {

        if (tensor_type()==TT_FULL) return full_tensor();
        else if (tensor_type()==TT_2D or tensor_type()==TT_TENSORTRAIN) return full_tensor_copy();
        else {
            MADNESS_EXCEPTION("you should not be here",1);
        }
        return Tensor<T>();
    }


    static double facReduce() {return 1.e-3;}
    static double fac_reduce() {return 1.e-3;}

    long rank() const {
        if (tensor_type()==TT_NONE) return 0;
        else if (tensor_type()==TT_FULL) return -1;
        else if (tensor_type()==TT_2D) return get_svdtensor().rank();
        else if (tensor_type()==TT_TENSORTRAIN) {
            std::vector<long> r=get_tensortrain().ranks();
            return *(std::max_element(r.begin(), r.end()));
        }
        else {
            MADNESS_EXCEPTION("you should not be here",1);
        }
        return 0l;
    }

    bool has_data() const {return (tensor_type() != TT_NONE);}

    bool has_no_data() const {return (not has_data());}

    long size() const {
    	return ptr()->size();
    }

    long real_size() const {
        if (tensor_type()==TT_NONE) return 0l;
        else if (tensor_type()==TT_FULL) return get_tensor().size();
        else if (tensor_type()==TT_2D) return get_svdtensor().real_size();
        else if (tensor_type()==TT_TENSORTRAIN) return get_tensortrain().real_size();
        else {
            MADNESS_EXCEPTION("you should not be here",1);
        }
        return false;
    }

    /// returns the trace of <this|rhs>
    template<typename Q>
    TENSOR_RESULT_TYPE(T,Q) trace_conj(const LowRankTensor<Q>& rhs) const {

        if (TensorTypeData<T>::iscomplex) MADNESS_EXCEPTION("no complex trace in LowRankTensor, sorry",1);
        if (TensorTypeData<Q>::iscomplex) MADNESS_EXCEPTION("no complex trace in LowRankTensor, sorry",1);

        typedef TENSOR_RESULT_TYPE(T,Q) resultT;
        // fast return if possible
        if ((this->rank()==0) or (rhs.rank()==0)) return resultT(0.0);

    	MADNESS_ASSERT(is_same_tensor_type(*this,rhs));

        if (tensor_type()==TT_FULL) return get_tensor().trace_conj(rhs.get_tensor());
        else if (tensor_type()==TT_2D) return trace(get_svdtensor(),rhs.get_svdtensor());
        else if (tensor_type()==TT_TENSORTRAIN) return get_tensortrain().trace(rhs.get_tensortrain());
        else {
            MADNESS_EXCEPTION("you should not be here",1);
        }
        return TENSOR_RESULT_TYPE(T,Q)(0);
    }

    /// multiply with a number
    template<typename Q>
    LowRankTensor<TENSOR_RESULT_TYPE(T,Q)> operator*(const Q& x) const {
        LowRankTensor<TENSOR_RESULT_TYPE(T,Q)> result(copy(*this));
        result.scale(x);
        return result;
    }

    LowRankTensor operator+(const LowRankTensor& other) {
    	LowRankTensor<T> result=copy(*this);
        result.gaxpy(1.0,other,1.0);
        return result;
    }

    LowRankTensor& operator+=(const LowRankTensor& other) {
        gaxpy(1.0,other,1.0);
        return *this;
    }

    LowRankTensor operator-(const LowRankTensor& other) {
    	LowRankTensor<T> result=copy(*this);
        result.gaxpy(1.0,other,-1.0);
        return result;
    }

    LowRankTensor& operator-=(const LowRankTensor& other) {
        gaxpy(1.0,other,-1.0);
        return *this;
    }

    LowRankTensor& gaxpy(const T alpha, const LowRankTensor& other, const T beta) {

    	// deliberately excluding gaxpys for different tensors due to efficiency considerations!
    	MADNESS_ASSERT(is_same_tensor_type(*this,other));
    	if (tensor_type()==TT_FULL) get_tensor().gaxpy(alpha,other.get_tensor(),beta);
    	else if (tensor_type()==TT_2D) get_svdtensor().gaxpy(alpha,other.get_svdtensor(),beta);
    	else if (tensor_type()==TT_TENSORTRAIN) get_tensortrain().gaxpy(alpha,other.get_tensortrain(),beta);
    	else {
    		MADNESS_EXCEPTION("unknown tensor type in LowRankTensor::gaxpy",1);
    	}
        return *this;
    }

    /// assign a number to this tensor
    LowRankTensor& operator=(const T& number) {
        std::visit([&number](auto& obj) {obj=number;}, tensor);
        return *this;
    }

    void add_SVD(const LowRankTensor& other, const double& thresh) {
        if (tensor_type()==TT_FULL) get_tensor()+=get_tensor();
        else if (tensor_type()==TT_2D) get_svdtensor().add_SVD(other.get_svdtensor(),thresh*facReduce());
        else if (tensor_type()==TT_TENSORTRAIN) get_tensortrain()+=(other.get_tensortrain());
        else {
    		MADNESS_EXCEPTION("unknown tensor type in LowRankTensor::add_SVD",1);
        }
    }

    /// Inplace multiply by corresponding elements of argument Tensor
    LowRankTensor<T>& emul(const LowRankTensor<T>& other) {

    	// deliberately excluding emuls for different tensors due to efficiency considerations!
    	MADNESS_ASSERT(is_same_tensor_type(*this,other));

        // binary operation with the visitor pattern
//        std::visit([&other](auto& obj) {obj.emul(other.tensor);}, tensor);
    	if (tensor_type()==TT_FULL) get_tensor().emul(other.get_tensor());
    	else if (tensor_type()==TT_2D) get_svdtensor().emul(other.get_svdtensor());
    	else if (tensor_type()==TT_TENSORTRAIN) get_tensortrain().emul(other.get_tensortrain());
    	else {
    		MADNESS_EXCEPTION("unknown tensor type in LowRankTensor::gaxpy",1);
    	}
        return *this;

    }

    void reduce_rank(const double& thresh) {

        if ((tensor_type()==TT_FULL) or (tensor_type()==TT_NONE)) return;
        else if (tensor_type()==TT_2D) get_svdtensor().divide_and_conquer_reduce(thresh*facReduce());
        else if (tensor_type()==TT_TENSORTRAIN) get_tensortrain().truncate(thresh*facReduce());
        else {
    		MADNESS_EXCEPTION("unknown tensor type in LowRankTensor::reduce_rank",1);
        }
    }


public:

    /// Transform all dimensions of the tensor t by the matrix c

    /// \ingroup tensor
    /// Often used to transform all dimensions from one basis to another
    /// \code
    /// result(i,j,k...) <-- sum(i',j', k',...) t(i',j',k',...) c(i',i) c(j',j) c(k',k) ...
    /// \endcode
    /// The input dimensions of \c t must all be the same and agree with
    /// the first dimension of \c c .  The dimensions of \c c may differ in
    /// size.
    template <typename R, typename Q>
    friend LowRankTensor<TENSOR_RESULT_TYPE(R,Q)> transform(
            const LowRankTensor<R>& t, const Tensor<Q>& c) {
    	typedef TENSOR_RESULT_TYPE(R,Q) resultT;
    	LowRankTensor<resultT> result;
        std::visit([&result, &c](auto& obj) {result=transform(obj,c);}, t.tensor);
        return result;
    }

    /// Transform all dimensions of the tensor t by distinct matrices c

    /// \ingroup tensor
    /// Similar to transform but each dimension is transformed with a
    /// distinct matrix.
    /// \code
    /// result(i,j,k...) <-- sum(i',j', k',...) t(i',j',k',...) c[0](i',i) c[1](j',j) c[2](k',k) ...
    /// \endcode
    /// The first dimension of the matrices c must match the corresponding
    /// dimension of t.    template <typename R, typename Q>
    template <typename R, typename Q>
    friend LowRankTensor<TENSOR_RESULT_TYPE(R,Q)> general_transform(
            const LowRankTensor<R>& t, const Tensor<Q> c[]) {
    	typedef TENSOR_RESULT_TYPE(R,Q) resultT;
    	LowRankTensor<resultT> result;
        std::visit([&result, &c](auto& obj) {result=general_transform(obj,c);}, t.tensor);
        return result;
    }

    /// Transforms one dimension of the tensor t by the matrix c, returns new contiguous tensor

    /// \ingroup tensor
    /// \code
    /// transform_dir(t,c,1) = r(i,j,k,...) = sum(j') t(i,j',k,...) * c(j',j)
    /// \endcode
    /// @param[in] t Tensor to transform (size of dim to be transformed must match size of first dim of \c c )
    /// @param[in] c Matrix used for the transformation
    /// @param[in] axis Dimension (or axis) to be transformed
    /// @result Returns a new, contiguous tensor    template <typename R, typename Q>
    template <typename R, typename Q>
    friend LowRankTensor<TENSOR_RESULT_TYPE(R,Q)> transform_dir(
            const LowRankTensor<R>& t, const Tensor<Q>& c, const int axis) {
    	LowRankTensor<TENSOR_RESULT_TYPE(R,Q)> result;
        std::visit([&result, &c, &axis](auto& obj) {result=transform_dir(obj,c,axis);}, t.tensor);
        return result;
    }

    template <typename R, typename Q>
    friend LowRankTensor<TENSOR_RESULT_TYPE(R,Q)> outer(
            const LowRankTensor<R>& t1, const LowRankTensor<Q>& t2);

    /// serialize this
    template <typename Archive>
    void serialize(Archive& ar) {

    	int index=tensor.index();
        ar & index;
        // index is now correct for load and store

        if (index==0) {
            MADNESS_ASSERT(std::holds_alternative<Tensor<T> >(tensor));
            ar & std::get<0>(tensor);
        }
        if (index==1) {
            MADNESS_ASSERT(std::holds_alternative<SVDTensor<T> >(tensor));
            ar & std::get<1>(tensor);
        }
        if (index==2) {
            MADNESS_ASSERT(std::holds_alternative<TensorTrain<T> >(tensor));
            ar & std::get<2>(tensor);
        }

    }

private:

    const BaseTensor* ptr() const {
        const BaseTensor* p;
        std::visit([&p](auto& obj) {p=dynamic_cast<const BaseTensor*>(&obj);}, tensor);
        return p;
    }

    /// holding the implementation of the low rank tensor representations
	std::variant<Tensor<T>, SVDTensor<T>, TensorTrain<T> > tensor;

};


/// type conversion implies a deep copy

/// @result Returns a new tensor that is a deep copy of the input
template <class Q, class T>
LowRankTensor<Q> convert(const LowRankTensor<T>& other) {

	// simple return
	if (std::is_same<Q, T>::value) return copy(other);

	LowRankTensor<Q> result;
    if (other.tensor_type()==TT_FULL)
        result=Tensor<Q>(convert<Q,T>(other.get_tensor()));
    if (other.tensor_type()==TT_2D)
        MADNESS_EXCEPTION("no type conversion for SVDTensors",1);
    if (other.tensor_type()==TT_TENSORTRAIN)
        MADNESS_EXCEPTION("no type conversion for TensorTrain",1);
    return result;
}


/// outer product of two Tensors, yielding a low rank tensor

/// do the outer product of two tensors; distinguish these tensortype cases by
/// the use of final_tensor_type
///  - full x full -> full
///  - full x full -> SVD                           ( default )
///  - TensorTrain x TensorTrain -> TensorTrain
/// all other combinations are currently invalid.
template <class T, class Q>
LowRankTensor<TENSOR_RESULT_TYPE(T,Q)> outer(const LowRankTensor<T>& t1,
        const LowRankTensor<Q>& t2, const TensorArgs final_tensor_args) {

    typedef TENSOR_RESULT_TYPE(T,Q) resultT;


    MADNESS_ASSERT(t1.tensor_type()==t2.tensor_type());

    if (final_tensor_args.tt==TT_FULL) {
        MADNESS_ASSERT(t1.tensor_type()==TT_FULL);
        Tensor<resultT> t(outer(t1.get_tensor(),t2.get_tensor()));
        return LowRankTensor<resultT>(t);

    } else if (final_tensor_args.tt==TT_2D) {
        MADNESS_ASSERT(t1.tensor_type()==TT_FULL);

        // srconf is shallow, do deep copy here
        const Tensor<T> lhs=t1.full_tensor_copy();
        const Tensor<Q> rhs=t2.full_tensor_copy();

        const long k=lhs.dim(0);
        const long dim=lhs.ndim()+rhs.ndim();
        long size=1;
        for (int i=0; i<lhs.ndim(); ++i) size*=k;
        MADNESS_ASSERT(size==lhs.size());
        MADNESS_ASSERT(size==rhs.size());
        MADNESS_ASSERT(lhs.size()==rhs.size());

        Tensor<double> weights(1);
        weights=1.0;

        SRConf<resultT> srconf(weights,lhs.reshape(1,lhs.size()),rhs.reshape(1,rhs.size()),dim,k);
//        srconf.normalize();
        return LowRankTensor<resultT>(SVDTensor<resultT>(srconf));

    } else if (final_tensor_args.tt==TT_TENSORTRAIN) {
        MADNESS_ASSERT(t1.tensor_type()==TT_TENSORTRAIN);
        MADNESS_ASSERT(t2.tensor_type()==TT_TENSORTRAIN);
        return outer(t1.get_tensortrain(),t2.get_tensortrain());
    } else {
        MADNESS_EXCEPTION("you should not be here",1);
    }
    return LowRankTensor<TENSOR_RESULT_TYPE(T,Q)>();

}

//
//namespace archive {
//    /// Serialize a tensor
//    template <class Archive, typename T>
//    struct ArchiveStoreImpl< Archive, LowRankTensor<T> > {
//
//        friend class LowRankTensor<T>;
//        /// Stores the GenTensor to an archive
//        static void store(const Archive& ar, const LowRankTensor<T>& t) {
//            bool exist=t.has_data();
//            int i=int(t.type);
//            ar & exist & i;
//            if (exist) {
//                if (t.impl.svd) ar & *t.impl.svd.get();
//                if (t.impl.full) ar & *t.impl.full.get();
//                if (t.impl.tt) ar & *t.impl.tt.get();
//            }
//        };
//    };
//
//
//    /// Deserialize a tensor ... existing tensor is replaced
//    template <class Archive, typename T>
//    struct ArchiveLoadImpl< Archive, LowRankTensor<T> > {
//
//        friend class GenTensor<T>;
//        /// Replaces this GenTensor with one loaded from an archive
//        static void load(const Archive& ar, LowRankTensor<T>& t) {
//            // check for pointer existence
//            bool exist=false;
//            int i=-1;
//            ar & exist & i;
//            t.type=TensorType(i);
//
//            if (exist) {
//                if (t.type==TT_2D) {
//                    SVDTensor<T> svd;
//                    ar & svd;
//                    t.impl.svd.reset(new SVDTensor<T>(svd));
//                } else if (t.type==TT_FULL) {
//                    Tensor<T> full;
//                    ar & full;
//                    t.impl.full.reset(new Tensor<T>(full));
//                } else if (t.type==TT_TENSORTRAIN) {
//                    TensorTrain<T> tt;
//                    ar & tt;
//                    t.impl.tt.reset(new TensorTrain<T>(tt));
//                }
//
//            }
//        };
//    };
//};


/// The class defines tensor op scalar ... here define scalar op tensor.
template <typename T, typename Q>
typename IsSupported < TensorTypeData<Q>, LowRankTensor<T> >::type
operator*(const Q& x, const LowRankTensor<T>& t) {
    return t*x;
}

/// add all the GenTensors of a given list

 /// If there are many tensors to add it's beneficial to do a sorted addition and start with
 /// those tensors with low ranks
 /// @param[in]  addends     a list with gentensors of same dimensions; will be destroyed upon return
 /// @param[in]  eps         the accuracy threshold
 /// @param[in]  are_optimal flag if the GenTensors in the list are already in SVD format (if TT_2D)
 /// @return     the sum GenTensor of the input GenTensors
 template<typename T>
 LowRankTensor<T> reduce(std::list<LowRankTensor<T> >& addends, double eps, bool are_optimal=false) {
     typedef typename std::list<LowRankTensor<T> >::iterator iterT;
     LowRankTensor<T> result=copy(addends.front());
      for (iterT it=++addends.begin(); it!=addends.end(); ++it) {
          result+=*it;
      }
      result.reduce_rank(eps);
      return result;

 }


/// implements a temporary(!) slice of a LowRankTensor
template<typename T>
class SliceLowRankTensor : public LowRankTensor<T> {
public:

    std::vector<Slice> thisslice;

    // all ctors are private, only accessible by GenTensor

    /// default ctor
    SliceLowRankTensor<T> () {}

    /// ctor with a GenTensor; shallow
    SliceLowRankTensor<T> (const LowRankTensor<T>& gt, const std::vector<Slice>& s)
    		: LowRankTensor<T>(gt), thisslice(s) {}

public:

    /// assignment as in g(s) = g1;
    SliceLowRankTensor<T>& operator=(const LowRankTensor<T>& rhs) {
        print("You don't want to assign to a SliceLowRankTensor; use operator+= instead");
        MADNESS_ASSERT(0);
        return *this;
    };

    /// assignment as in g(s) = g1(s);
    SliceLowRankTensor<T>& operator=(const SliceLowRankTensor<T>& rhs) {
        print("You don't want to assign to a SliceLowRankTensor; use operator+= instead");
        MADNESS_ASSERT(0);
        return *this;
    };

    /// inplace addition as in g(s)+=g1
    SliceLowRankTensor<T>& operator+=(const LowRankTensor<T>& rhs) {

        // fast return if possible
        if (rhs.has_no_data() or rhs.rank()==0) return *this;

        // no fast return possible!!!
        //          if (this->rank()==0) {
        //              // this is a deep copy
        //              *this=rhs(rhs_s);
        //              this->scale(beta);
        //              return;
        //          }

        std::vector<Slice> rhs_slice(rhs.ndim(),Slice(_));

        if (this->tensor_type()==TT_FULL) {
            this->get_tensor()(thisslice).gaxpy(1.0,rhs.get_tensor(),1.0);

        } else if (this->tensor_type()==TT_2D) {
            this->get_svdtensor().inplace_add(rhs.get_svdtensor(),thisslice,rhs_slice, 1.0, 1.0);

        } else if (this->tensor_type()==TT_TENSORTRAIN) {
            this->get_tensortrain().gaxpy(thisslice,rhs.get_tensortrain(),1.0,rhs_slice);
        }
        return *this;
    }

    /// inplace subtraction as in g(s)-=g1
    SliceLowRankTensor<T>& operator-=(const LowRankTensor<T>& rhs) {

        // fast return if possible
        if (rhs.has_no_data() or rhs.rank()==0) return *this;

        // no fast return possible!!!
        //          if (lrt.rank()==0) {
        //              // this is a deep copy
        //              *this=rhs(rhs_s);
        //              lrt.scale(beta);
        //              return;
        //          }

        std::vector<Slice> rhs_slice(rhs.ndim(),Slice(_));

        if (this->tensor_type()==TT_FULL) {
            (this->impl.full)(thisslice).gaxpy(1.0,(rhs.get_tensor())(rhs_slice),-1.0);

        } else if (this->tensor_type()==TT_2D) {
            this->impl.svd->inplace_add(rhs.get_svdtensor(),thisslice,rhs_slice, 1.0, -1.0);

        } else if (this->tensor_type()==TT_TENSORTRAIN) {
            this->impl.tt->gaxpy(thisslice,rhs.get_tensortrain(),-1.0,rhs_slice);
        }
        return *this;
    }

    /// inplace addition as in g(s)+=g1(s)
    SliceLowRankTensor<T>& operator+=(const SliceLowRankTensor<T>& rhs) {
        // fast return if possible
        if (rhs.has_no_data() or rhs.rank()==0) return *this;

        if (this->has_data()) MADNESS_ASSERT(this->tensor_type()==rhs.tensor_type());

        // no fast return possible!!!
        //          if (lrt.rank()==0) {
        //              // this is a deep copy
        //              *this=rhs(rhs_s);
        //              lrt.scale(beta);
        //              return;
        //          }

        std::vector<Slice> rhs_slice=rhs.thisslice;

        if (this->tensor_type()==TT_FULL) {
            this->get_tensor()(thisslice).gaxpy(1.0,(rhs.get_tensor())(rhs_slice),1.0);

        } else if (this->tensor_type()==TT_2D) {
            this->get_svdtensor().inplace_add(rhs.get_svdtensor(),thisslice,rhs_slice, 1.0, 1.0);

        } else if (this->tensor_type()==TT_TENSORTRAIN) {
            this->get_tensortrain().gaxpy(thisslice,rhs.get_tensortrain(),1.0,rhs_slice);
        }
        return *this;

    }

    /// inplace zero-ing as in g(s)=0.0
    SliceLowRankTensor<T>& operator=(const T& number) {
        MADNESS_ASSERT(number==T(0.0));

        if (this->tensor_type()==TT_FULL) {
            this->get_tensor()(thisslice)=0.0;

        } else if (this->tensor_type()==TT_2D) {
            MADNESS_ASSERT(this->get_svdtensor().has_structure());
            LowRankTensor<T> tmp(*this);
            this->get_svdtensor().inplace_add(tmp.get_svdtensor(),thisslice,thisslice, 1.0, -1.0);

        } else if (this->tensor_type()==TT_TENSORTRAIN) {
            this->get_tensortrain().gaxpy(thisslice,this->get_tensortrain(),-1.0,thisslice);
        }
        return *this;
    }

    friend LowRankTensor<T> copy(const SliceLowRankTensor<T>& other) {
        LowRankTensor<T> result;
        const std::vector<Slice> s=other.thisslice;
        if (result.tensor_type()==TT_FULL)
            result=Tensor<T>(copy((other.get_tensor())(s)));
        else if (result.tensor_type()==TT_2D)
            result=SVDTensor<T>(other.get_svdtensor().copy_slice(s));
        else if (result.tensor_type()==TT_TENSORTRAIN)
            result=TensorTrain<T>(copy(other.get_tensortrain(),s));
        else {
            MADNESS_EXCEPTION("you should not be here",1);
        }
        return result;

    }


};



} // namespace madness

#endif /* MADNESS_TENSOR_LOWRANKTENSOR_H_ */
