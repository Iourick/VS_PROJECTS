﻿
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



using namespace std;

char strInpFolder[] = "..//FDMT_TESTS//2048";
char strPathOutImageNpyFile_gpu[] = "out_image_GPU.npy";



//extern int IROWS = 0;
//extern int ICOLS = 0;
//extern  std::vector<std::vector<int>> ivctOut = std::vector<std::vector<int>>(1, std::vector<int>(1, 0));


//--------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
	//--------------------------------------------------------------------------------------
	
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
	
		
	free(piarr);	
	cudaFree(d_piarrImage);
	cudaFree(d_piarrState0);
	cudaFree(d_piarrState0);
	cudaFree(d_arr_val0);
	cudaFree(d_arr_val1);
	cudaFree(d_arr_deltaTLocal);
	cudaFree(d_arr_dT_MI);
	cudaFree(d_arr_dT_ML);
	cudaFree(d_arr_dT_RI);

	char filename_cu[] = "image_GPU.png";
	createImg_(argc, argv, v1, IImgcols, IMaxDT, filename_cu);

	return 0;
}
//-------------------------------------------------------------------------
//void fncFdmt_cu_v0(int* piarrImgInp, const int IImgrows, const int IImgcols
//	, const float VAlFmin, const  float VAlFmax, const int IMaxDT, int* piarrImgOut)
//{
//	 1. quant iteration's calculation
//	const int I_F = (int)(log2((double)(IImgrows)));
//	 !1
//
//	 2. temp variables calculations
//	const float val_dF = (VAlFmax - VAlFmin) / ((float)(IImgrows));
//	int ideltaT = int(ceil((IMaxDT - 1.) * (1. / (VAlFmin * VAlFmin)
//		- 1. / ((VAlFmin + val_dF) * (VAlFmin + val_dF)))
//		/ (1. / (VAlFmin
//			* VAlFmin) - 1. / (VAlFmax * VAlFmax))));
//	 !2
//
//
//	 3. declare pointers to device arrays
//	int* d_p0 = 0;
//	int* d_p1 = 0;
//	int* d_piarrOut_0 = 0;
//	int* d_piarrOut_1 = 0;
//	int* d_piarrImgInp = 0;
//	 !3
//
//	 4. allocate memory to device arrays
//	clock_t start = clock();
//	cudaMalloc(&d_piarrOut_0, IImgrows * (ideltaT + 1) * IImgcols * sizeof(int));
//	cudaMalloc(&d_piarrOut_1, IImgrows * (ideltaT + 1) * IImgcols * sizeof(int));
//	cudaMalloc(&d_piarrImgInp, IImgrows * IImgcols * sizeof(int));
//	 !4
//	clock_t end = clock();
//	double duration = double(end - start) / CLOCKS_PER_SEC;
//	std::cout << "Time taken by cudaMalloc: " << duration << " seconds" << std::endl;
//
//	 5  Initialize the device arrays with zeros
//	start = clock();
//	cudaMemset(d_piarrOut_0, 0, IImgrows * (ideltaT + 1) * IImgcols * sizeof(int));
//	cudaMemset(d_piarrOut_1, 0, IImgrows * (ideltaT + 1)/2 * IImgcols * sizeof(int));
//	 !5
//	end = clock();
//	duration = double(end - start) / CLOCKS_PER_SEC;
//	std::cout << "Time taken by cudaMemset: " << duration << " seconds" << std::endl;
//
//	 6.copy input data from host to device
//	start = clock();
//	cudaMemcpy(d_piarrImgInp, piarrImgInp, IImgrows * IImgcols * sizeof(int)
//		, cudaMemcpyHostToDevice);
//	end = clock();
//	duration = double(end - start) / CLOCKS_PER_SEC;
//	std::cout << "Time taken by cudaMemcpy: " << duration << " seconds" << std::endl;
//	 !6
//
//	 7. call initialization func
//	start = clock();
//	fnc_init(d_piarrImgInp, IImgrows, IImgcols, ideltaT, d_piarrOut_0);
//	fnc_init_fdmt(d_piarrImgInp, IImgrows, IImgcols, ideltaT, d_piarrOut_0);
//	end = clock();
//	duration = double(end - start) / CLOCKS_PER_SEC;
//	std::cout << "Time taken by fnc_init: " << duration << " seconds" << std::endl;
//	 !7
//
//	/*start = clock()*1000.;
//	for (int i = 0; i < 1000; ++i)
//	{
//		cudaMemcpy(d_piarrImgInp, d_piarrOut_0, IImgrows * IImgcols * sizeof(int)
//			, cudaMemcpyDeviceToDevice);
//	}
//	end = clock() * 1000.;
//	duration = double(end - start) / CLOCKS_PER_SEC;
//	std::cout << "Time taken by MemcpyDeviceToDevice: " << duration << " microseconds" << std::endl;
//
//	start = clock() * 1000.;
//	for (int i = 0; i < IImgrows; ++i)
//	{
//		cudaMemcpy(d_piarrImgInp, d_piarrOut_0, IImgrows  * sizeof(int)
//			, cudaMemcpyDeviceToDevice);
//	}
//	end = clock() * 1000.;
//	duration = double(end - start) / CLOCKS_PER_SEC;
//	std::cout << "Time taken by MemcpyDeviceToDevice 1 row x IImgrowstimes  : " << duration << " miliseconds" << std::endl;*/
//
//	 8.pointers initialization
//	d_p0 = d_piarrOut_0;
//	d_p1 = d_piarrOut_1;
//	 8!
//
//	 9. allocate memory to device  auxiliary arrays
//	start = clock();
//	float* d_arr_val0 = 0;
//	cudaMalloc(&d_arr_val0, IImgrows / 2 * sizeof(float));
//
//	float* d_arr_val1 = 0;
//	cudaMalloc(&d_arr_val1, IImgrows / 2 * sizeof(float));
//
//	int* d_arr_deltaTLocal = 0;
//	cudaMalloc(&d_arr_deltaTLocal, IImgrows / 2 * sizeof(int));
//
//	int* d_arr_dT_MI = 0;
//	cudaMalloc(&d_arr_dT_MI, IImgrows * (ideltaT + 1) * sizeof(int));
//
//	int* d_arr_dT_ML = 0;
//	cudaMallocManaged(&d_arr_dT_ML, IImgrows * (ideltaT + 1) * sizeof(int));
//
//	int* d_arr_dT_RI = 0;
//	cudaMalloc(&d_arr_dT_RI, IImgrows * (ideltaT + 1) * sizeof(int));
//
//	end = clock();
//	duration = double(end - start) / CLOCKS_PER_SEC;
//	std::cout << "Time taken by allocating memory to device  auxiliary arrays: " << duration << " seconds" << std::endl;
//	 !9
//	int iInp0 = IImgrows;
//	int iInp1 = ideltaT + 1;
//
//	int iOut0 = 0, iOut1 = 0, iOut2 = 0;
//
//	 10. iterations
//	start = clock();
//	for (int iit = 1; iit < (I_F + 1); ++iit)
//	{
//		fncFdmtIteration(d_p0, val_dF, iInp0, iInp1
//			, IImgcols, IMaxDT, VAlFmin
//			, VAlFmax, iit, d_arr_val0
//			, d_arr_val1, d_arr_deltaTLocal
//			, d_arr_dT_MI, d_arr_dT_ML, d_arr_dT_RI
//			, d_p1, iOut0, iOut1);
//
//		 exchange order of pointers
//		int* d_pt = d_p0;
//		d_p0 = d_p1;
//		d_p1 = d_pt;
//		iInp0 = iOut0;
//		iInp1 = iOut1;
//
//		 !
//	}
//	end = clock();
//	duration = double(end - start) / CLOCKS_PER_SEC;
//	std::cout << "Time taken by iterations: " << duration << " seconds" << std::endl;
//	 ! 10
//
//	 11. copy output to host
//	start = clock();
//	cudaMemcpy(piarrImgOut, d_p0, IImgcols * IMaxDT
//		* sizeof(int), cudaMemcpyDeviceToHost);
//	end = clock();
//	duration = double(end - start) / CLOCKS_PER_SEC;
//	std::cout << "Time taken by copying memory to host: " << duration << " seconds" << std::endl;
//	 ! 11
//
//	 12 free memory on device
//	start = clock();
//	cudaFree(d_arr_val0);
//	cudaFree(d_arr_val1);
//	cudaFree(d_arr_deltaTLocal);
//
//	cudaFree(d_piarrOut_0);
//	cudaFree(d_piarrOut_1);
//	cudaFree(d_piarrImgInp);
//
//	cudaFree(d_arr_dT_MI);
//	cudaFree(d_arr_dT_ML);
//	cudaFree(d_arr_dT_RI);
//	cudaDeviceReset();
//
//	end = clock();
//	duration = double(end - start) / CLOCKS_PER_SEC;
//	std::cout << "Time taken by cudaFree: " << duration << " seconds" << std::endl;
//	 !12
//
//}
//
//--------------------------------------------------------------------------------------
//    Input :
//    Input - 3d array, with dimensions[N_f, N_d0, Nt]
//    f_min, f_max - are the base - band begin and end frequencies.
//    The frequencies can be entered in both MHz and GHz, units are factored out in all uses.
//    maxDT - the maximal delay(in time bins) of the maximal dispersion.
//    Appears in the paper as N_{\Delta}
//A typical input is maxDT = N_f
//dataType - To naively use FFT, one must use floating point types.
//Due to casting, use either complex64 or complex128.
//iteration num - Algorithm works in log2(Nf) iterations, each iteration changes all the sizes(like in FFT)
//Output:
//3d array, with dimensions[N_f / 2, N_d1, Nt]
//    where N_d1 is the maximal number of bins the dispersion curve travels at one output frequency band
//
//    For details, see algorithm 1 in Zackay & Ofek(2014)
// F,T = Image.shape 
// d_piarrInp имеет  размерности IDim0, IDim1,IDim2
// IDim0: this is iImgrows - quantity of rows of input power image, this is F
// IDim1: changes 
// IDim2: this is iImgcols - quantity of cols of input power image, this is T 
//void fncFdmtIteration(int* d_piarrInp, const float val_dF, const int IDim0, const int IDim1
//	, const int IDim2, const int IMaxDT, const float VAlFmin
//	, const float VAlFmax, const int ITerNum, float* d_arr_val0
//	, float* d_arr_val1, int* d_iarr_deltaTLocal
//	, int* d_iarr_dT_MI, int* d_iarr_dT_ML, int* d_iarr_dT_RI
//	, int* d_piarrOut, int& iOutPutDim0, int& iOutPutDim1)
//{
//
//	float valDeltaF = pow(2., ITerNum) * val_dF;
//	float temp0 = 1. / (VAlFmin * VAlFmin) -
//		1. / ((VAlFmin + valDeltaF) * (VAlFmin + valDeltaF));
//
//	const float VAlTemp1 = 1. / (VAlFmin * VAlFmin) - 1. / (VAlFmax * VAlFmax);
//
//	int ideltaT = (int)(ceil(((float)IMaxDT - 1.0) * temp0 / VAlTemp1));
//	iOutPutDim1 = ideltaT + 1;
//	iOutPutDim0 = IDim0 / 2;
//
//
//	 set zeros in output array
//	cudaMemset(d_piarrOut, 0, iOutPutDim0 * iOutPutDim1 * IDim2 * sizeof(int));
//	 !
//
//	float val_correction = 0;
//	if (ITerNum > 0)
//	{
//		val_correction = val_dF / 2.;
//	}
//
//
//	 9. auxiliary constants initialization
//	const float VAlC2 = (VAlFmax - VAlFmin) / ((float)(iOutPutDim0));
//	 !9	
//
//	 10. calculating first 3 auxillary 1 dim arrays
//	int threadsPerBlock = 1024;
//	int numberOfBlocks = (iOutPutDim0 + threadsPerBlock - 1) / threadsPerBlock;
//
//	create_auxillary_1d_arrays << <numberOfBlocks, threadsPerBlock >> > (iOutPutDim0
//		, IMaxDT, VAlTemp1, VAlC2, VAlFmin, val_correction
//		, d_arr_val0, d_arr_val1, d_iarr_deltaTLocal);
//	cudaDeviceSynchronize();
//
//	 !10
//
//
//
//	 12. calculating second 3 auxillary 2 dim arrays
//	int quantEl = iOutPutDim0 * iOutPutDim1;
//	threadsPerBlock = 256;
//	numberOfBlocks = (quantEl + threadsPerBlock - 1) / threadsPerBlock;
//
//	kernel_2d_arrays << < numberOfBlocks, threadsPerBlock >> > (iOutPutDim0
//		, iOutPutDim1, d_arr_val0, d_arr_val1, d_iarr_deltaTLocal
//		, d_iarr_dT_MI, d_iarr_dT_ML
//		, d_iarr_dT_RI);
//	cudaDeviceSynchronize();
//
//	 !11
//
//	 13. 
//	quantEl = iOutPutDim0 * iOutPutDim1 * IDim2;
//	numberOfBlocks = (quantEl + threadsPerBlock - 1) / threadsPerBlock;
//	kernel_shift_and_sum << <numberOfBlocks, threadsPerBlock >> > (d_piarrInp
//		, IDim0, IDim1, IDim2, d_iarr_deltaTLocal, d_iarr_dT_MI, d_iarr_dT_ML, d_iarr_dT_RI
//		, iOutPutDim0, iOutPutDim1, d_piarrOut);
//	/*shift_and_sum (d_piarrInp
//		, IDim0, IDim1, IDim2, d_iarr_deltaTLocal, d_iarr_dT_MI, d_iarr_dT_ML, d_iarr_dT_RI
//		, iOutPutDim0, iOutPutDim1, d_piarrOut);*/
//	cudaDeviceSynchronize();
//
//
//
//}
//
//-----------------------------------------------------------------------------------------------------------------------
//__global__
//void kernel_shift_and_sum(int* d_piarrInp, const int IDim0, const int IDim1
//	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
//	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
//	, int* d_piarrOut)
//{
//	const int i = blockIdx.x * blockDim.x + threadIdx.x;
//	if (i >= IOutPutDim0 * IOutPutDim1 * IDim2)
//	{
//		return;
//	}
//	int iw = IOutPutDim1 * IDim2;
//	int i_F = i / iw;
//	int irest = i % iw;
//	int i_dT = irest / IDim2;
//	if (i_dT > d_iarr_deltaTLocal[i_F])
//	{
//		return;
//	}
//	int idx = irest % IDim2;
//	 claculation of bound index: 
//	 arr_dT_ML[i_F, i_dT]
//	 index of arr_dT_ML
//	 arr_dT_ML is matrix with IOutPutDim0 rows and IOutPutDim1 cols
//	int ind = i_F * IOutPutDim1 + i_dT;
//	 !
//
//	 calculation of:
//	 d_Output[i_F][i_dT][idx] = d_input[2 * i_F][arr_dT_MI[i_F, i_dT]][idx]
//	   calculation num row of submatix No_2 * i_F of d_piarrInp = arr_dT_MI[ind]
//	d_piarrOut[i] = d_piarrInp[2 * i_F * IDim1 * IDim2 +
//		d_iarr_dT_MI[ind] * IDim2 + idx];
//
//	if (idx >= d_iarr_dT_ML[ind])
//	{
//		int numRow = d_iarr_dT_RI[ind];
//		int indInpMtrx = (2 * i_F + 1) * IDim1 * IDim2 + numRow * IDim2 + idx - d_iarr_dT_ML[ind];
//		atomicAdd(&d_piarrOut[i], d_piarrInp[ind]);
//		d_piarrOut[i] += d_piarrInp[indInpMtrx];
//	}
//}
//
//-----------------------------------------------------------------------------------------------------------------------
//
//void shift_and_sum(int* d_piarrInp, const int IDim0, const int IDim1
//	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
//	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
//	, int* d_piarrOut)
//{
//	int iarr_deltaTLocal[1000] = { 0 };
//	cudaMemcpy(iarr_deltaTLocal, d_iarr_deltaTLocal, IOutPutDim0 * sizeof(int)
//		, cudaMemcpyDeviceToHost);
//
//	int* iarr_dT_ML = (int*)malloc(IOutPutDim0 * IOutPutDim1 * sizeof(int));
//	cudaMemcpy(iarr_dT_ML, d_iarr_dT_ML, IOutPutDim0 * IOutPutDim1 * sizeof(int)
//		, cudaMemcpyDeviceToHost);
//
//	int* iarr_dT_MI = (int*)malloc(IOutPutDim0 * IOutPutDim1 * sizeof(int));
//	cudaMemcpy(iarr_dT_MI, d_iarr_dT_MI, IOutPutDim0 * IOutPutDim1 * sizeof(int)
//		, cudaMemcpyDeviceToHost);
//
//	int* iarr_dT_RI = (int*)malloc(IOutPutDim0 * IOutPutDim1 * sizeof(int));
//	cudaMemcpy(iarr_dT_RI, d_iarr_dT_RI, IOutPutDim0 * IOutPutDim1 * sizeof(int)
//		, cudaMemcpyDeviceToHost);
//
//
//	for (int i_F = 0; i_F < IOutPutDim0; ++i_F)
//	{
//
//		for (int i_dT = 0; i_dT < (1 + iarr_deltaTLocal[i_F]); ++i_dT)
//		{
//			int numRowOutputMtrxBegin0 = i_F * IOutPutDim1 * IDim2 + i_dT * IDim2;
//			 number of element of beginning of the input 2 * i_F matrix's row with number 
//			 dT_middle_index[i_F][i_dT]
//			int numRowInputMtrxBegin0 = 2 * i_F * IDim1 * IDim2 + IDim2 * (iarr_dT_MI[i_F * IOutPutDim1 + i_dT]);
//			cudaMemcpy(&d_piarrOut[numRowOutputMtrxBegin0], &d_piarrInp[numRowInputMtrxBegin0], IDim2 * sizeof(int)
//				, cudaMemcpyDeviceToDevice);
//
//			 number of beginning element of summated rows
//			int numElemInRow = iarr_dT_ML[i_F * IOutPutDim1 + i_dT];
//			 number of beginning element of output matrix  Output[i_F, i_dT, dT_middle_larger:]
//			int numRowOutputMtrxBegin1 = numRowOutputMtrxBegin0 + numElemInRow;
//
//			 number of the row of the submatrix of input matrix with number 2 * i_F + 1
//			int numRowOfInputSubmatrix = iarr_dT_RI[i_F * IOutPutDim1 + i_dT];
//			 number of beginning element of the input matrix Input[2 * i_F + 1, dT_rest_index, :i_T_max - dT_middle_larger]
//			int numRowInputMtrxBegin1 = (2 * i_F + 1) * IDim1 * IDim2 + IDim2 * numRowOfInputSubmatrix;
//			int threadsPerBlock = 1024;
//			int numberOfBlocks = (IDim2 - numElemInRow + threadsPerBlock - 1) / threadsPerBlock;
//			sumArrays_ << <numberOfBlocks, threadsPerBlock >> > (&d_piarrOut[numRowOutputMtrxBegin1], &d_piarrInp[numRowInputMtrxBegin1], IDim2 - numElemInRow);
//			cudaDeviceSynchronize();
//		}
//	}
//	free(iarr_dT_RI);
//	free(iarr_dT_ML);
//	free(iarr_dT_MI);
//}
//__global__
//void create_auxillary_1d_arrays(const int IFjumps, const int IMaxDT, const float VAlTemp1
//	, const float VAlc2, const float VAlf_min, const float VAlcorrection
//	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal)
//{
//	const int i = blockIdx.x * blockDim.x + threadIdx.x;
//	if (i > IFjumps)
//	{
//		return;
//	}
//	float valf_start = VAlc2 * i + VAlf_min;
//	float valf_end = valf_start + VAlc2;
//	float valf_middle_larger = VAlc2 / 2. + valf_start + VAlcorrection;
//	float valf_middle = VAlc2 / 2. + valf_start - VAlcorrection;
//	float temp0 = 1. / (valf_start * valf_start) - 1. / (valf_end * valf_end);
//
//	d_arr_val0[i] = -(1. / (valf_middle * valf_middle) - 1. / (valf_start * valf_start)) / temp0;
//
//	d_arr_val1[i] = -(1. / (valf_middle_larger * valf_middle_larger)
//		- 1. / (valf_start * valf_start)) / temp0;
//
//	d_iarr_deltaTLocal[i] = (int)(ceil((((float)(IMaxDT)) - 1.) * temp0 / VAlTemp1));
//}
//--------------------------------------------------------------------------------------
//__global__
//void kernel_2d_arrays(const int IDim0, const int IDim1
//	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal
//	, int* d_iarr_dT_middle_index, int* d_iarr_dT_middle_larger
//	, int* d_iarr_dT_rest_index)
//
//{
//	const int i = blockIdx.x * blockDim.x + threadIdx.x;
//
//	if (i > IDim0 * IDim1)
//	{
//		return;
//	}
//	int i_F = i / IDim1;
//	int i_dT = i % IDim1;
//	if (i_dT > (d_iarr_deltaTLocal[i_F]))
//	{
//		d_iarr_dT_middle_index[i] = 0;
//		d_iarr_dT_middle_larger[i] = 0;
//		d_iarr_dT_rest_index[i] = 0;
//		return;
//	}
//
//	d_iarr_dT_middle_index[i] = round(((float)i_dT) * d_arr_val0[i_F]);
//	int ivalt = round(((float)i_dT) * d_arr_val1[i_F]);
//	d_iarr_dT_middle_larger[i] = ivalt;
//	d_iarr_dT_rest_index[i] = i_dT - ivalt;
//
//
//}
//--------------------------------------------------------------------------------------
//void fnc_init(int* d_piarrImg, const int IImgrows, const int IImgcols
//	, const int IDeltaT, int* d_piarrOut)
//{
//	// output in .npy:
//		int* parr = (int*)malloc(IImgrows * IImgcols  * sizeof(int));
//		cudaMemcpy(parr, d_piarrImg, IImgrows * IImgcols  * sizeof(int)
//			, cudaMemcpyDeviceToHost);
//		std::vector<int> v6(parr, parr + IImgrows * IImgcols );
//
//		std::array<long unsigned, 1> leshape126 {IImgrows* IImgcols};
//
//		npy::SaveArrayAsNumpy("init00.npy", false, leshape126.size(), leshape126.data(), v6);
//		free(parr);
//
//	cudaMemset(d_piarrOut, 0, IImgrows * IImgcols * (IDeltaT + 1) * sizeof(int));
//
//	for (int i = 0; i < IImgrows; ++i)
//	{
//		{
//			cudaMemcpy(&d_piarrOut[i * (IDeltaT + 1) * IImgcols], &d_piarrImg[i * IImgcols]
//				, IImgcols * sizeof(int), cudaMemcpyDeviceToDevice);
//		}
//	}
//
//
//	for (int i_dT = 1; i_dT < (IDeltaT + 1); ++i_dT)
//		for (int iF = 0; iF < IImgrows; ++iF)
//		{
//
//			int threadsPerBlock = 1024;
//			int numberOfBlocks = (IImgcols - i_dT + threadsPerBlock - 1) / threadsPerBlock;
//			int* d_result = &d_piarrOut[iF * (IDeltaT + 1) * IImgcols + i_dT * IImgcols + i_dT];
//			int* d_arg0 = &d_piarrOut[iF * (IDeltaT + 1) * IImgcols + (i_dT - 1) * IImgcols + i_dT];
//			int* d_arg1 = &d_piarrImg[iF * IImgcols];
//			sumArrays << <numberOfBlocks, threadsPerBlock >> > (d_result, d_arg0, d_arg1, IImgcols - i_dT);
//			cudaDeviceSynchronize();
//		}
//
//}
//
//-----------------------------------------------------------------------------
//CUDA kernel for element-wise summation
//__global__ void sumArrays(int* d_result, const int* d_arr1, const int* d_arr2, int n)
//{
//	int tid = blockIdx.x * blockDim.x + threadIdx.x;
//	if (tid < n)
//	{
//		d_result[tid] = d_arr1[tid] + d_arr2[tid];
//	}
//}
//-----------------------------------------------------------------------------
//CUDA kernel for element-wise summation
//__global__ void sumArrays_(int* d_result, const int* d_arr1, int n)
//{
//	int tid = blockIdx.x * blockDim.x + threadIdx.x;
//	if (tid < n)
//	{
//		d_result[tid] += d_arr1[tid];
//	}
//}
//
//
//--------------------------------------------------------------------------------------
//void fnc_init_fdmt(int* d_piarrImg, const int IImgrows, const int IImgcols
//	, const int IDeltaT, int* d_piarrOut)
//{
//	// output in .npy:
//		int* parr = (int*)malloc(IImgrows * IImgcols  * sizeof(int));
//		cudaMemcpy(parr, d_piarrImg, IImgrows * IImgcols  * sizeof(int)
//			, cudaMemcpyDeviceToHost);
//		std::vector<int> v6(parr, parr + IImgrows * IImgcols );
//
//		std::array<long unsigned, 1> leshape126 {IImgrows* IImgcols};
//
//		npy::SaveArrayAsNumpy("init00.npy", false, leshape126.size(), leshape126.data(), v6);
//		free(parr);
//
//	int threadsPerBlock = 1024;
//	int numberOfBlocks = (IImgrows* IImgcols + threadsPerBlock - 1) / threadsPerBlock;
//	kernel_seed<<< numberOfBlocks, threadsPerBlock>>>(d_piarrImg, IImgrows, IImgcols
//		, IDeltaT, d_piarrOut);
//	cudaDeviceSynchronize();
//
//	for (int i_dT = 1; i_dT < (IDeltaT + 1); ++i_dT)
//	{
//		init_iter << < numberOfBlocks, threadsPerBlock >> > (d_piarrImg, IImgrows, IImgcols, IDeltaT
//			, i_dT, d_piarrOut);
//		cudaDeviceSynchronize();
//	}
//	}		
//	
//---------------------------------------------------------------------------
//__global__
//void kernel_seed(int* d_piarrImg, const int IImgrows, const int IImgcols
//	, const int IDeltaT, int* d_piarrOut)
//{
//	const int IDeltaTplus1 = IDeltaT + 1;
//	int tid = blockIdx.x * blockDim.x + threadIdx.x;
//	if (tid > IImgrows * IImgcols)
//	{
//		return;
//	}
//	int quantElemInMtrx = (IImgcols * IDeltaTplus1);
//	int iF = tid / IImgcols;
//	int numElemInRow = tid % IImgcols;
//
//	d_piarrOut[iF * quantElemInMtrx + numElemInRow] = d_piarrImg[iF * IImgcols + numElemInRow];
//
//}
//---------------------------------------------------------------------------------------------------
//__global__
// void init_iter(int* d_piarrImg, const int IImgrows, const int IImgcols, const int IDeltaT
//, const int i_dT, int* d_piarrOut)
//{
//
//	int tid = blockIdx.x * blockDim.x + threadIdx.x;
//	if (tid > IImgrows * IImgcols)
//	{
//		return;
//	}
//	const int IDeltaTplus1 = IDeltaT + 1;
//	int quantElemInMtrx = (IImgcols * IDeltaTplus1);
//	int iF = tid / IImgcols;
//	int numElemInRow = tid % IImgcols;
//	int* ip0 = &d_piarrOut[iF * quantElemInMtrx + (i_dT-1) * IImgcols + numElemInRow];
//	int* ip1 = ip0 + IImgcols;
//
//
//	if (i_dT <= numElemInRow)
//	{
//		*ip1 = *ip0 + d_piarrImg[iF * IImgcols + numElemInRow - i_dT];
//			d_piarrOut[iF * quantElemInMtrx + (i_dT - 1) * IImgcols + numElemInRow] + d_piarrImg[iF * IImgcols + numElemInRow - i_dT];
//		/*atomicAdd(&d_piarrOut[iF * quantElemInMtrx + i_dT * IImgcols + numElemInRow],
//			d_piarrImg[iF * IImgcols + numElemInRow - i_dT]);*/
//	}
//	else
//	{
//		*ip1 = 0;
//	}
//
//	
//}
//
//-------------------------------------------------------------------------------------------------------------------------------
//void fncCalcDimensionsOfOutputArrays(std::vector<int>* pivctOutDim0, std::vector<int>* pivctOutDim1
//, std::vector<int>* pivctOutDim2,  const int IDim0, const int IDim1
//, const int IDim2,  const int IMaxDT, const float VAlFmin
//, const float VAlFmax)
//{
//	float val_dF = (VAlFmax - VAlFmin) / ((float)((*pivctOutDim0)[0]));
//	
//	
//	
//	for (int it = 1; it < pivctOutDim0->size(); ++it)
//	{
//		float valDeltaF = pow(2., it) * val_dF;
//		float temp0 = 1. / (VAlFmin * VAlFmin) -
//			1. / ((VAlFmin + valDeltaF) * (VAlFmin + valDeltaF));
//		const float VAlTemp1 = 1. / (VAlFmin * VAlFmin) -
//			1. / (VAlFmax * VAlFmax);
//		int ideltaT = (int)(ceil(((float)IMaxDT - 1.0) * temp0 / VAlTemp1));
//		(*pivctOutDim1)[it] = ideltaT + 1;
//		(*pivctOutDim0)[it] = (*pivctOutDim0)[it-1] / 2;
//		(*pivctOutDim2)[it] = (*pivctOutDim2)[it-1];
//
//		
//	}
//}
//

