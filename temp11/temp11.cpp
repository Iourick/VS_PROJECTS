// temp11.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include "npy.hpp"
#include <vector>
#include <string>

int main() {
	const std::vector<long unsigned> shape{ 2, 3 };
	const bool fortran_order{ false };
	const std::string path{ "out.npy" };

	const std::vector<double> data{ 1, 2, 3, 4, 5, 6 };
	npy::SaveArrayAsNumpy(path, fortran_order, shape.size(), shape.data(), data);


	std::vector<unsigned long> shape1{};
	//bool fortran_order;
	std::vector<double> data1;

	//const std::string path{ "data.npy" };
	//npy::LoadArrayFromNumpy(path, shape1, fortran_order, data1);
	npy::LoadArrayFromNumpy(path, shape1,  data1);

	int gg = 0;
}
//LoadArrayFromNumpy(const std::string& filename, std::vector<unsigned long>& shape, bool& fortran_order,
//	std::vector<Scalar>& data)
//
//	LoadArrayFromNumpy(const std::string& filename, std::vector<unsigned long>& shape,
//		std::vector<Scalar>& data)