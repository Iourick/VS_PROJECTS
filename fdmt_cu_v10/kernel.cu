
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "FdmtCu1.cuh"

#include <math.h>
#include <stdio.h>
#include <array>
#include <iostream>
#include <string>

#include <vector>
#include <cstdlib> // For random value generation
#include <ctime>   // For seeding the random number generator
#include "npy.hpp"
#include <algorithm> 
#include "kernel.cuh"
#include <chrono>
#include "fileInput.h"
#include "DrawImg.h"



using namespace std;

char strInpFolder[] = "..//FDMT_TESTS//2048";
char strPathOutImageNpyFile_gpu[] = "out_image_GPU.npy";



//extern int IROWS = 0;
//extern int ICOLS = 0;
//extern  std::vector<std::vector<int>> ivctOut = std::vector<std::vector<int>>(1, std::vector<int>(1, 0));


//--------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
	
	//const std::vector<int> data1 {1, 2, 3, 4, 5, 6};
	//std::array<long unsigned, 2> leshape11 {2, 3};
	//std::array<long unsigned, 1> leshape12 {6};

	//const double data2[]{ 7 };
	//std::array<long unsigned, 3> leshape21 {1, 1, 1};
	//std::array<long unsigned, 0> leshape22 {};

	//const std::array<double, 0> data3;
	//std::array<long unsigned, 2> leshape31 {4, 0};

	//npy::SaveArrayAsNumpy("out11.npy", false, leshape11.size(), leshape11.data(), data1);
	//npy::SaveArrayAsNumpy("out12.npy", false, leshape12.size(), leshape12.data(), data1);

	//npy::SaveArrayAsNumpy("out21.npy", false, leshape21.size(), leshape21.data(), data2);
	//npy::SaveArrayAsNumpy("out22.npy", false, leshape22.size(), leshape22.data(), data2);

	//npy::SaveArrayAsNumpy("out31.npy", false, leshape31.size(), leshape31.data(), data3.data());

	//std::vector<unsigned long> sh {};
	//std::vector<int> vctD;
	//bool bf = false;
	//npy::LoadArrayFromNumpy("out12.npy", sh, bf, vctD);

	//int n = 5;  // Specify the length of the array
	//int iarr[] = { 1, 2, 3, 4, 9 };  // Your one-dimensional integer array

	//// Create a std::vector from the integer array
	//std::vector<int> v(iarr, iarr + n);
	//-------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//------------------- prepare to work -------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
// initiate pointer to input image
	int* piarr = (int*)malloc(sizeof(int));

	// initiate 2-pointer to input image, in order to realloc memory to satisfy arbitrary dimensions
	int** ppiarrImage = &piarr;

	// initiating input variables
	int iMaxDT = 0;
	int iImRows = 0, iImCols = 0;
	float val_fmax = 0., val_fmin = 0.;

	// reading input files from folder 
	int ireturn = downloadInputData(strInpFolder, &iMaxDT, ppiarrImage, &iImRows, &iImCols,
		&val_fmin, &val_fmax);

	// analysis output of reading function
	switch (ireturn)
	{
	case 1:
		cout << "Err. ...Can't allocate memory for input image. Oooops... " << std::endl;
		return 1;
	case 2:
		cout << "Err. ...Input dimensions must be a power of 2. Oooops... " << std::endl;
		return 1;
	case 0:
		cout << "Input data downloaded properly " << std::endl;
		break;
	default:
		cout << "..something extraordinary happened! Oooops..." << std::endl;
		break;
	}

	// declare constants
	const int IMaxDT = iMaxDT;
	const int IImgrows = iImRows;
	const int IImgcols = iImCols;
	const float VAlFmin = val_fmin;
	const float VAlFmax = val_fmax;


	// handy pointer to input image
	int* piarrImage = *ppiarrImage;

	

	//--------------------------------------------------------------------------------------------------------------
	//-------------------- end of prepare ------------------------------------------------------------------------------------------
	//------------------- begin to calculate cuda var -------------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------------------------------
	// 1. allocate memory for device array
	

	int* u_piarrImOut = 0;
	cudaMallocManaged(&u_piarrImOut, IImgcols * IMaxDT * sizeof(int));

	int* d_piarrImage = 0;
	//cudaMallocManaged(&d_piarrImage, IImgcols * IImgrows * sizeof(int));
	cudaMalloc(&d_piarrImage, IImgcols * IImgrows * sizeof(int));

	
	// !1


	// 1. quant iteration's calculation
	const int I_F = (int)(log2((double)(IImgrows)));
	// !1

	// 2. temp variables calculations
	const float VAl_dF = (VAlFmax - VAlFmin) / ((float)(IImgrows));
	const int IDeltaT = int(ceil((IMaxDT - 1.) * (1. / (VAlFmin * VAlFmin)
		- 1. / ((VAlFmin + VAl_dF) * (VAlFmin + VAl_dF)))
		/ (1. / (VAlFmin
			* VAlFmin) - 1. / (VAlFmax * VAlFmax))));
	// !2


	// 3. declare pointers to device arrays
	
	int* d_piarrState0 = 0;
	int* d_piarrState1 = 0;
	
	// !3

	// 4. allocate memory to device arrays
	
	cudaMalloc(&d_piarrState0, IImgrows * (IDeltaT + 1) * IImgcols * sizeof(int));
	cudaMalloc(&d_piarrState1, IImgrows * (IDeltaT + 1) * IImgcols * sizeof(int));
	
	// !4
	
	// 5  Initialize the device arrays with zeros
	
	cudaMemset(d_piarrState0, 0, IImgrows * (IDeltaT + 1) * IImgcols * sizeof(int));
	cudaMemset(d_piarrState1, 0, IImgrows * (IDeltaT + 1) / 2 * IImgcols * sizeof(int));
	// !5

	// 9. allocate memory to device  auxiliary arrays
	
	float* d_arr_val0 = 0;
	cudaMalloc(&d_arr_val0, IImgrows / 2 * sizeof(float));

	float* d_arr_val1 = 0;
	cudaMalloc(&d_arr_val1, IImgrows / 2 * sizeof(float));

	int* d_arr_deltaTLocal = 0;
	cudaMalloc(&d_arr_deltaTLocal, IImgrows / 2 * sizeof(int));

	int* d_arr_dT_MI = 0;
	cudaMalloc(&d_arr_dT_MI, IImgrows * (IDeltaT + 1) * sizeof(int));

	int* d_arr_dT_ML = 0;
	cudaMallocManaged(&d_arr_dT_ML, IImgrows * (IDeltaT + 1) * sizeof(int));

	int* d_arr_dT_RI = 0;
	cudaMalloc(&d_arr_dT_RI, IImgrows * (IDeltaT + 1) * sizeof(int));
	


	// 2. calculations	
	//clock_t start = clock();	
	auto start = std::chrono::high_resolution_clock::now();

	
	fncFdmt_cu_v0(piarrImage  // input image
		, d_piarrImage 
		, IImgrows, IImgcols // dimensions of input image 	
		, d_piarrState0		// auxillary allocated buffer of mrmory in device
		, d_piarrState1		// auxillary allocated buffer of mrmory in device
		, IDeltaT
		, I_F
		, VAl_dF
		, d_arr_val0 		 // auxillary allocated buffer of mrmory in device
		, d_arr_val1			// auxillary allocated buffer of mrmory in device
		, d_arr_deltaTLocal	 // auxillary allocated buffer of mrmory in device
		, d_arr_dT_MI			// auxillary allocated buffer of mrmory in device
		, d_arr_dT_ML			// auxillary allocated buffer of mrmory in device
		, d_arr_dT_RI			// auxillary allocated buffer of mrmory in device
		, VAlFmin, VAlFmax, IMaxDT, u_piarrImOut);
	
		auto end = std::chrono::high_resolution_clock::now();

	// Вычисляем разницу времени
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	// Выводим время выполнения в микросекундах
	std::cout << "Time taken by function fncFdmt_cu_v0: " << duration.count() << " milliseconds" << std::endl;
	// !2
	
	// output in .npy:IImgcols * IMaxDT * sizeof(int));
	/*int * piarrImOut = (int*)malloc(IImgcols * IMaxDT * sizeof(int));
	cudaMemcpy(piarrImOut, u_piarrImOut, IImgcols* IMaxDT * sizeof(int), cudaMemcpyDeviceToHost);*/
	std::vector<int> v1(u_piarrImOut, u_piarrImOut + IImgcols * IMaxDT);

	std::array<long unsigned, 2> leshape101 {IImgcols , IMaxDT};

	npy::SaveArrayAsNumpy(strPathOutImageNpyFile_gpu, false, leshape101.size(), leshape101.data(), v1);

	
	
	//--------------------------------------------------------------------------------------------------------------
	//-------------------- end of calculations ------------------------------------------------------------------------------------------
	//------------------- begin to draw output image for cuda -------------------------------------------------------------------------------------------
	
	char filename_cu[] = "image_GPU.png";
	createImg( argc, argv, u_piarrImOut, IImgcols, IMaxDT, filename_cu);
	
	free(piarr);
	cudaFree(u_piarrImOut);
	//free(piarrImOut);
	cudaFree(d_piarrImage);
	cudaFree(d_piarrState0);
	cudaFree(d_piarrState0);
	cudaFree(d_arr_val0);
	cudaFree(d_arr_val1);
	cudaFree(d_arr_deltaTLocal);
	cudaFree(d_arr_dT_MI);
	cudaFree(d_arr_dT_ML);
	cudaFree(d_arr_dT_RI);

	return 0;
}
