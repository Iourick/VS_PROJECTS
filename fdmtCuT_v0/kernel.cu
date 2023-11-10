
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "FdmtCu0.cuh"

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

#define  TILE_DIM  16



using namespace std;

char strInpFolder[] = "..//FDMT_TESTS//2048";
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
__global__ void transposeMatrix(float* input, float* output, int width, int height) {
	__shared__ float tile[TILE_DIM][TILE_DIM];
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;

	if (x < width && y < height) {
		tile[threadIdx.y][threadIdx.x] = input[y * width + x];
	}
	__syncthreads();

	x = blockIdx.y * TILE_DIM + threadIdx.x;
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	if (x < height && y < width) {
		output[y * height + x] = tile[threadIdx.x][threadIdx.y];
		//output[y * height + x] = tile[threadIdx.x][threadIdx.y];
	}
}


//--------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
	const int width = 512;  // Width of the matrix
	const int height = 512; // Height of the matrix
	 // Tile size
	const int matrixSize = width * height;

	// Allocate memory for the input and output matrices on the host (CPU).
	float* h_inputMatrix = (float*)malloc(matrixSize * sizeof(float));
	float* h_outputMatrix = (float*)malloc(matrixSize * sizeof(float));

	// Initialize the input matrix with your data.
	// Don't forget to populate h_inputMatrix with the input data.
	float* d_inputMatrix;
	float* d_outputMatrix;
	cudaMalloc((void**)&d_inputMatrix, matrixSize * sizeof(float));
	cudaMalloc((void**)&d_outputMatrix, matrixSize * sizeof(float));

	// Copy the input matrix from host to device memory.
	cudaMemcpy(d_inputMatrix, h_inputMatrix, matrixSize * sizeof(float), cudaMemcpyHostToDevice);
	dim3 dimBlock(TILE_DIM, TILE_DIM); // Thread block size
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y); // Grid size
	int num11 = 1000;
	auto start11 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < num11; ++i)
	{

		transposeMatrix << <dimGrid, dimBlock >> > (d_inputMatrix, d_outputMatrix, width, height);
	}
	cudaDeviceSynchronize(); // Wait for the kernel to finish
	auto end11 = std::chrono::high_resolution_clock::now();
	auto duration11 = std::chrono::duration_cast<std::chrono::microseconds>(end11 - start11);
	std::cout << "Time taken by transposeMatrix: " << duration11.count() / ((double)num11) << " microseconds" << std::endl;
	cudaMemcpy(h_outputMatrix, d_outputMatrix, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);
	free(h_inputMatrix);
	free(h_outputMatrix);

//--------------------------------------------------------------------------

	printDeviceInfo();
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

	// 5.1
	if ((iImRows == 1024) && (BDIM_512_1024))
	{
		iImRows = 512;
		iMaxDT = 512;
		piarr = (int*)realloc(piarr, iImRows * iImCols * sizeof(int));

	}
	// ! 5.1

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
	
	int num = 1;
	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < num; ++i)
	{
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
	}
	
	auto end = std::chrono::high_resolution_clock::now();	
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);	
	std::cout << "Time taken by function fncFdmt_cu_v0: " << duration.count() /((double)num)<< " microseconds" << std::endl;
	// !2
	
	// output in .npy:IImgcols * IMaxDT * sizeof(int));
	int * piarrImOut = (int*)malloc(IImgcols * IMaxDT * sizeof(int));
	cudaMemcpy(piarrImOut, u_piarrImOut, IImgcols* IMaxDT * sizeof(int), cudaMemcpyDeviceToHost);
	std::vector<int> v1(piarrImOut, piarrImOut + IImgcols * IMaxDT);

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



	free(piarr);
	cudaFree(d_piarrImage);
	cudaFree(d_piarrState0);
	cudaFree(d_piarrState1);
	cudaFree(d_arr_val0);
	cudaFree(d_arr_val1);
	cudaFree(d_arr_deltaTLocal);
	cudaFree(d_arr_dT_MI);
	cudaFree(d_arr_dT_ML);
	cudaFree(d_arr_dT_RI);
	free(piarrImOut);

	char filename_cpu[] = "image_cpu.png";
	createImg_(argc, argv, v1, IImgcols, IMaxDT, filename_cpu);

	return 0;
}
