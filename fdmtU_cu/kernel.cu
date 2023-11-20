
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "fdmtU_cu.cuh"

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
#include "Constants.h"





using namespace std;

char strInpFolder[] = "..//FDMT_TESTS//512";
char strPathOutImageNpyFile_gpu[] = "out_image_GPU.npy";
const bool BDIM_512_1024 = true;


void printDeviceInfo()
{
	int deviceId;
	int numberOfSMs;
	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
	cudaDeviceProp deviceProps;
	cudaGetDeviceProperties(&deviceProps, deviceId);
	std::string deviceName = deviceProps.name;
	std::cout << "Device Name: " << deviceName << std::endl;
	std::cout << "Number of SM: " << numberOfSMs << std::endl;
}


//---------------------------------------

int main(int argc, char** argv)
{

	
	printDeviceInfo();
//--------------------------------------------------------------------------------------------------------------
//------------------- prepare to work -------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
// initiate pointer to input image
	int * piarr = (int *)malloc(sizeof(int));

	// initiate 2-pointer to input image, in order to realloc memory to satisfy arbitrary dimensions
	int ** ppiarrImage = &piarr;

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

	// transform data to float
	float* parrImage = (float*)malloc(iImRows * iImCols * sizeof(float));
	for (int i = 0; i < iImRows * iImCols; ++i)
	{
		parrImage[i] = (float)(piarr[i]);
	}
	free(piarr);
	// !
	

	// 5.1
	if ((iImRows == 1024) && (BDIM_512_1024))
	{
		iImRows = 512;
		iMaxDT = 512;
		parrImage = (float*)realloc(parrImage, iImRows * iImCols * sizeof(float));

	}
	// ! 5.1

	// declare constants
	const int IMaxDT = iMaxDT;
	const int IImgrows = iImRows;
	const int IImgcols = iImCols;
	const float VAlFmin = val_fmin;
	const float VAlFmax = val_fmax;


	

	

	//--------------------------------------------------------------------------------------------------------------
	//-------------------- end of prepare ------------------------------------------------------------------------------------------
	//------------------- begin to calculate cuda var -------------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------------------------------
	// 1. allocate memory for device array
	

	float* u_parrImOut = 0;
	cudaMallocManaged(&u_parrImOut, IImgcols * IMaxDT * sizeof(float));

	float* d_parrImage = 0;
	
	cudaMalloc(&d_parrImage, IImgcols * IImgrows * sizeof(float));

	
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
	
	float* d_parrState0 = 0;
	float* d_parrState1 = 0;
	
	// !3

	// 4. allocate memory to device arrays
	
	cudaMalloc(&d_parrState0, IImgrows * (IDeltaT + 1) * IImgcols * sizeof(float));
	cudaMalloc(&d_parrState1, IImgrows * (IDeltaT + 1) * IImgcols * sizeof(float));
	
	// !4
	
	// 5  Initialize the device arrays with zeros
	
	cudaMemset(d_parrState0, 0, IImgrows * (IDeltaT + 1) * IImgcols * sizeof(float));
	cudaMemset(d_parrState1, 0, IImgrows * (IDeltaT + 1) / 2 * IImgcols * sizeof(float));
	// !5

	// 6. allocate memory to device  auxiliary arrays
	
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
	
	int num = 1000;
	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < num; ++i)
	{
		fncFdmtU_cu(parrImage  // input image
			, d_parrImage
			, IImgrows, IImgcols // dimensions of input image 	
			, d_parrState0		// auxillary allocated buffer of mrmory in device
			, d_parrState1		// auxillary allocated buffer of mrmory in device
			, IDeltaT
			, I_F
			, VAl_dF
			, d_arr_val0 		 // auxillary allocated buffer of mrmory in device
			, d_arr_val1			// auxillary allocated buffer of mrmory in device
			, d_arr_deltaTLocal	 // auxillary allocated buffer of mrmory in device
			, d_arr_dT_MI			// auxillary allocated buffer of mrmory in device
			, d_arr_dT_ML			// auxillary allocated buffer of mrmory in device
			, d_arr_dT_RI			// auxillary allocated buffer of mrmory in device
			, VAlFmin, VAlFmax, IMaxDT, u_parrImOut);
	}
	
	auto end = std::chrono::high_resolution_clock::now();	
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);	
	std::cout << "Time taken by function fncFdmt_cu_v0: " << duration.count() /((double)num)<< " microseconds" << std::endl;
	// !2
	
	// output in .npy:IImgcols * IMaxDT * sizeof(int));
	float * parrImOut = (float*)malloc(IImgcols * IMaxDT * sizeof(float));
	cudaMemcpy(parrImOut, u_parrImOut, IImgcols* IMaxDT * sizeof(float), cudaMemcpyDeviceToHost);
	std::vector<int> v1(parrImOut, parrImOut + IImgcols * IMaxDT);

	std::array<long unsigned, 2> leshape101 {IImgcols , IMaxDT};

	npy::SaveArrayAsNumpy(strPathOutImageNpyFile_gpu, false, leshape101.size(), leshape101.data(), v1);

	
	
	//--------------------------------------------------------------------------------------------------------------
	//-------------------- end of calculations ------------------------------------------------------------------------------------------
	//------------------- begin to draw output image for cuda -------------------------------------------------------------------------------------------
	
	float flops = 0;
	if (iImRows == 512)
	{
		flops = GFLPS_512;
	}
	else
	{
		if (iImRows == 1024)
		{
			if (BDIM_512_1024)
			{
				flops = GFLPS_512_1024;
			}
			else
			{
				flops = GFLPS_1024;
			}
		}
		else
		{
			flops = GFLPS_2048;
		}
	}

	cout << "GFLP/sec = " << ((double)flops) / ((double)duration.count() / ((double)num)) * 1.0e6  << endl;

	/*int deviceId;
	int numberOfSMs;	
	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
	cudaDeviceProp deviceProps;
	cudaGetDeviceProperties(&deviceProps, deviceId);
	std::string deviceName = deviceProps.name;
	std::cout << "Device Name: " << deviceName << std::endl;
	std::cout << "Number of SM: " << numberOfSMs << std::endl;*/



	
	cudaFree(d_parrImage);
	cudaFree(d_parrState0);
	cudaFree(d_parrState1);
	cudaFree(d_arr_val0);
	cudaFree(d_arr_val1);
	cudaFree(d_arr_deltaTLocal);
	cudaFree(d_arr_dT_MI);
	cudaFree(d_arr_dT_ML);
	cudaFree(d_arr_dT_RI);
	free(parrImOut);

	free(parrImage);

	char filename_cpu[] = "image_cpu.png";
	createImg_(argc, argv, v1, IImgcols, IMaxDT, filename_cpu);

	return 0;
}
