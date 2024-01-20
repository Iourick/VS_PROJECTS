//THIS FILE CONTAINS FUNCTION TO READ FOUR <*>.NPY FILES WITH INPUT INFORMATION

#define _CRT_SECURE_NO_WARNINGS
#include "fileInput.h"
#include <math.h>
#include <iostream>
#include "npy.hpp"

using namespace std;


int downloadInputData(char* strFolder,  int* iMaxDT, int** ppiarrImage
	, int* iImRows, int* iImCols, float*  val_fmin, float* val_fmax)
{
	bool fortran_order = false;
	// 1. loading typeofdata
	std::vector<unsigned long> shape {};
	std::vector<int> imaxDT;
	char arrch0[] = "//imaxDT.npy";
	char chpath0[100] = { 0 };
	strcpy(chpath0, strFolder);
	strcat(chpath0, arrch0);
	npy::LoadArrayFromNumpy(chpath0, shape, fortran_order, imaxDT);
	*iMaxDT = imaxDT[0];
	// !1

	// 2. loading XX
	char arrch1[] = "//XX.npy";
	char chpath1[100] = { 0 };
	strcpy(chpath1, strFolder);
	strcat(chpath1, arrch1);
	std::vector<unsigned long> shape1 {};
	std::vector<int> vctXX;
	npy::LoadArrayFromNumpy(chpath1, shape1, fortran_order, vctXX);
	// !2

	// 3. loading shape
	char arrch2[] = "//iarrShape.npy";
	char chpath2[100] = { 0 };
	strcpy(chpath2, strFolder);
	strcat(chpath2, arrch2);
	std::vector<unsigned long> shape0 {};
	std::vector<int> ivctImShape;
	npy::LoadArrayFromNumpy(chpath2, shape0, fortran_order, ivctImShape);
	*iImRows = ivctImShape[0];
	*iImCols = ivctImShape[1];
	// !3

	// 4. loading fmin and fmax 
	char arrch3[] = "//fmin_max.npy";
	char chpath3[100] = { 0 };
	strcpy(chpath3, strFolder);
	strcat(chpath3, arrch3);
	std::vector<unsigned long> shape2 {};
	std::vector<float> vctfmin_max;
	npy::LoadArrayFromNumpy(chpath3, shape2, fortran_order, vctfmin_max);
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