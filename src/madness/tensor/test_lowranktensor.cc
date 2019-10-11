/*
 * LRTensor_test.cc
 *
 *  Created on: Oct 2, 2019
 *      Author: fbischoff
 */

#include "lowranktensor.h"


using namespace madness;

template<typename T>
double compute_difference(const LowRankTensor<T>& lrt1, const LowRankTensor<T>& lrt2) {\
	return (lrt1.full_tensor_copy()-lrt2.full_tensor_copy()).normf();
}

template<typename T>
double compute_difference(const LowRankTensor<T>& lrt1, const Tensor<T>& t2) {\
	return (lrt1.full_tensor_copy()-t2).normf();
}

template<typename T>
double compute_difference(const Tensor<T>& t2, const LowRankTensor<T>& lrt1) {\
	return (lrt1.full_tensor_copy()-t2).normf();
}

std::vector<long> make_dimensions() {
	madness::Random(10);

	long ndim=(madness::RandomValue<long>() %5+2);	// anything from 2 to 6
	std::vector<long> dim(ndim);
	for (long& d : dim) d=(madness::RandomValue<long>() %7+1);// anything from 1 to 7
	print("dimensions",dim);
	return dim;
}


template<typename T>
int test_constructor() {
	print("\nentering test_construct");
	std::vector<long> dim=make_dimensions();
	Tensor<T> tensor(dim);
	double thresh=1.e-5;
	for (int i=0; i<2; ++i) {
		print("repeating with random tensors");
		SVDTensor<T> svd(tensor,thresh);
		TensorTrain<T> tt(tensor,thresh);

		LowRankTensor<T> lrt;
		LowRankTensor<T> lrt1(tensor);
		LowRankTensor<T> lrt2(tensor,thresh,TT_2D);
		LowRankTensor<T> lrt3(tensor,thresh,TT_TENSORTRAIN);
		LowRankTensor<T> lrt4(tensor,thresh,TT_FULL);
		LowRankTensor<T> lrt5(lrt4);

		LowRankTensor<T> lrt6(svd);
		LowRankTensor<T> lrt7(tt);

		LowRankTensor<T> lrt8(dim,TT_FULL);
		LowRankTensor<T> lrt9(dim,TT_2D);
		LowRankTensor<T> lrt10(dim,TT_TENSORTRAIN);
		LowRankTensor<T> lrt11(TT_2D,dim[0],dim.size());
		tensor.fillrandom();

	}

	// not throwing is success
	return 0;
}

template<typename T>
int test_reconstruct() {
	print("\nentering test_reconstruct");

	std::vector<long> dim=make_dimensions();
	Tensor<T> tensor(dim);
	tensor.fillrandom();
	double thresh=1.e-5;
	double error=0.0;

	LowRankTensor<T> lrt1(tensor);
	LowRankTensor<T> lrt2(tensor,thresh,TT_2D);
	LowRankTensor<T> lrt3(tensor,thresh,TT_TENSORTRAIN);
	LowRankTensor<T> lrt4(tensor,thresh,TT_FULL);

	double error1=compute_difference(lrt1,tensor);
	double error2=compute_difference(lrt2,tensor);
	double error3=compute_difference(lrt3,tensor);
	double error4=compute_difference(lrt4,tensor);

	print("error1",error1);
	print("error2",error2);
	print("error3",error3);
	print("error4",error4);

	error=error1+error2+error3+error4;
	return 0;
}



template<typename T>
int test_addition(const TensorType& tt) {

	print("\nentering test_addition", tt);
	std::vector<long> dim=make_dimensions();
	Tensor<T> tensor(dim);
	tensor.fillrandom();
	double error=0.0;

	LowRankTensor<T> lrt1(tensor,TensorArgs(1.e-4,tt));
	LowRankTensor<T> lrt2(tensor,TensorArgs(1.e-4,tt));
	Tensor<T> tensor2=copy(tensor);
	lrt2+=lrt1;
	tensor2+=tensor;
	double error1=compute_difference(lrt2,tensor2);
	print("error1",error1);
	error+=error1;

	Tensor<T> tensor3=tensor+tensor;
	LowRankTensor<T> lrt3=lrt1+lrt1;
	double error2=compute_difference(lrt3,tensor3);
	print("error2",error2);
	error+=error2;

	Tensor<T> tensor4=tensor-tensor2;
	LowRankTensor<T> lrt4=lrt1-lrt2;
	double error4=compute_difference(lrt4,tensor4);
	print("error4",error4);
	error+=error4;

	if (tt==TT_2D) {
		tensor4+=tensor2;
		lrt4.reduce_rank(1.e-4);
		lrt2.reduce_rank(1.e-4);
		lrt4.get_svdtensor().add_SVD(lrt2.get_svdtensor(),1.e-4);
		double error5=compute_difference(lrt4,tensor4);
		print("error5",error5);
		error+=error5;


	}


	return (error>1.e-8);
}


template<typename T>
int test_sliced_assignment(const TensorType& tt) {

	print("\nentering test_addition", tt);
	Tensor<T> tensor(10,10,10,10);
	tensor.fillrandom();
	double error=0.0;
	std::vector<Slice> s0={_,_,Slice(0,8),Slice(3,5)};
	std::vector<Slice> s1={_,_,Slice(1,9),Slice(2,4)};

	return (error>1.e-8);
}

template<typename T>
int test_sliced_addition(const TensorType& tt) {

	print("\nentering test_slicing_addition", tt);
	Tensor<T> tensor1(10,8,11,10);
	Tensor<T> tensor2=3.0*copy(tensor1);
	tensor1.fillrandom();
	double error=0.0;
	std::vector<Slice> s1={_,_,Slice(0,8),Slice(3,5)};
	std::vector<Slice> s2={_,_,Slice(1,9),Slice(2,4)};

	LowRankTensor<T> lrt1(tensor1,TensorArgs(1.e-4,tt));
	LowRankTensor<T> lrt2(tensor2,TensorArgs(1.e-4,tt));

	tensor1(s1)+=tensor2(s2);
	lrt1(s1)+=lrt2(s2);
	double error2=compute_difference(lrt1,tensor1);
	print("error2",error2);
	error+=error2;


	return (error>1.e-8);
}


template<typename T>
int test_reduce_rank(const TensorType& tt) {

	print("\nentering test_reduce_rank", tt);
	std::vector<long> dim=make_dimensions();
	Tensor<T> tensor1(dim);
	Tensor<T> tensor2(dim);
	tensor1.fillrandom();
	tensor2.fillrandom();
	double error=0.0;
	double thresh=1.e-5;

	LowRankTensor<T> lrt1(tensor1,TensorArgs(1.e-4,tt));
	LowRankTensor<T> lrt2(tensor2,TensorArgs(1.e-4,tt));

	tensor1+=tensor2;
	lrt1+=lrt2;
	lrt1.reduce_rank(thresh);

	double error2=compute_difference(lrt1,tensor1);
	print("error2",error2);
	error+=error2;


	return (error>1.e-8);
}



template<typename T>
int test_emul(const TensorType& tt) {

	print("\nentering test_emul", tt);
	std::vector<long> dim=make_dimensions();
	Tensor<T> tensor1(dim);
	Tensor<T> tensor2(dim);
	tensor1.fillrandom();
	tensor2.fillrandom();
	double error=0.0;
	double thresh=1.e-5;

	LowRankTensor<T> lrt1(tensor1,TensorArgs(1.e-4,tt));
	LowRankTensor<T> lrt2(tensor2,TensorArgs(1.e-4,tt));

	tensor1.emul(tensor2);
	lrt1.emul(lrt2);

	double error2=compute_difference(lrt1,tensor1);
	print("error2",error2);
	error+=error2;


	return (error>1.e-8);
}

template<typename T>
int test_convert() {
	print("\nentering test_convert");

	Tensor<T> tensor(10,10,10,10);
	tensor.fillrandom();
	double thresh=1.e-5;

	LowRankTensor<T> lrt(tensor);
	LowRankTensor<T> lrt1=lrt.convert(TensorArgs(TT_2D,thresh));
	LowRankTensor<T> lrt2=lrt.convert(TensorArgs(TT_TENSORTRAIN,thresh));

	LowRankTensor<T> lrt3=lrt1.convert(TensorArgs(TT_FULL,thresh));
	LowRankTensor<T> lrt4=lrt2.convert(TensorArgs(TT_FULL,thresh));

	LowRankTensor<T> lrt5=lrt2.convert(TensorArgs(TT_2D,thresh));

	double error1=compute_difference(lrt1,tensor);
	double error2=compute_difference(lrt2,tensor);
	double error3=compute_difference(lrt3,tensor);
	double error4=compute_difference(lrt4,tensor);
	double error5=compute_difference(lrt5,tensor);
	print("error1", error1);
	print("error2", error2);
	print("error3", error3);
	print("error4", error4);
	print("error5", error5);

	return 0;

}

template<typename T>
int test_general_transform() {
	print("\nentering test_general_transform");

	Tensor<T> tensor(10,10,10,10);
	Tensor<T> c(10,10);
	tensor.fillrandom();
	c.fillrandom();
	double thresh=1.e-5;
	Tensor<double> cc[TENSOR_MAXDIM];
	for (long idim=0; idim<4; idim++) {
		cc[idim]=Tensor<double>(10,10);
		cc[idim].fillrandom();
	}

	LowRankTensor<T> lrt1(tensor);
	LowRankTensor<T> lrt2(tensor,thresh,TT_TENSORTRAIN);
	LowRankTensor<T> lrt3(tensor,thresh,TT_2D);
	LowRankTensor<T> result1=general_transform(lrt1,cc);
	LowRankTensor<T> result2=general_transform(lrt2,cc);
	LowRankTensor<T> result3=general_transform(lrt3,cc);

	double error2=(result2.full_tensor_copy()-result1.full_tensor_copy()).normf();
	double error3=(result2.full_tensor_copy()-result1.full_tensor_copy()).normf();
	print("error2",error2);
	print("error3",error3);

	return 0;
}

int
main(int argc, char* argv[]) {

	madness::default_random_generator.setstate(int(cpu_time())%4149);

    int success = test_constructor<double>();
    success += test_constructor<double_complex>();

//    success +=  test_reconstruct<double>();
//    success +=  test_reconstruct<double_complex>();

    success +=  test_convert<double>();
    success +=  test_convert<double_complex>();

    std::vector<TensorType> tt={TT_FULL,TT_2D,TT_TENSORTRAIN};
    for (auto& t : tt) {
    	success += test_addition<double>(t);
    	success += test_sliced_addition<double>(t);
    	success += test_reduce_rank<double>(t);
    	success += test_emul<double>(t);
    }

    success += test_general_transform<double>();
    success += test_general_transform<double_complex>();


    std::cout << "Test " << ((success==0) ? "passed" : "did not pass") << std::endl;

    return success;
}
