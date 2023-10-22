#include "fileInput.h"
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib> // For random value generation
#include "npy.hpp"

using namespace std;
std::vector<std::vector<int>> ivctOut;

int downloadInputData(char* strFolder, char* strOutImageNpyFile, int* iMaxDT, int** ppiarrImage
	, int* iImRows, int* iImCols, float*  val_fmin, float* val_fmax)
{
	bool fortran_order = false;
	// 1. loading typeofdata
	std::vector<unsigned long> shape {};
	std::vector<int> imaxDT;
	npy::LoadArrayFromNumpy("D://MyVSprojPy//imaxDT.npy", shape, fortran_order, imaxDT);
	*iMaxDT = imaxDT[0];
	// !1

	// 2. loading XX
	std::vector<unsigned long> shape1 {};
	std::vector<int> vctXX;
	npy::LoadArrayFromNumpy("D://MyVSprojPy//XX.npy", shape1, fortran_order, vctXX);
	// !2

	// 3. loading shape
	std::vector<unsigned long> shape0 {};
	std::vector<int> ivctImShape;
	npy::LoadArrayFromNumpy("D://MyVSprojPy//iarrShape.npy", shape0, fortran_order, ivctImShape);
	*iImRows = ivctImShape[0];
	*iImCols = ivctImShape[1];
	// !3

	// 4. loading fmin and fmax  
	std::vector<unsigned long> shape2 {};
	std::vector<float> vctfmin_max;
	npy::LoadArrayFromNumpy("D://MyVSprojPy//fmin_max.npy", shape2, fortran_order, vctfmin_max);
	*val_fmin = vctfmin_max[0];
	*val_fmax = vctfmin_max[1];
	// ! 4

	// 5.checking dimensions
	bool bDim0_OK = false, bDim1_OK = false;
	int numcur = 2;
	for (int i = 1; i < 31; ++i)
	{
		numcur = 2 * numcur;
		if (numcur == ivctImShape[0])
		{
			bDim0_OK = true;
		}
		if (numcur == ivctImShape[1])
		{
			bDim1_OK = true;
		}
	}
	if (!(bDim0_OK && bDim1_OK)) {

		return 2;
	}
	// ! 5

	// 6. realloc and fill array Image
	
	size_t size = (size_t)(vctXX.size() * sizeof(int));
	
	if (!(*ppiarrImage = (int*)realloc(*ppiarrImage, size)))
	{
		return 1;
	}
	
	for (int i = 0; i < vctXX.size(); ++i)
	{
		(*ppiarrImage)[i] = vctXX[i];

	}
	// !6
	
	return 0;

}