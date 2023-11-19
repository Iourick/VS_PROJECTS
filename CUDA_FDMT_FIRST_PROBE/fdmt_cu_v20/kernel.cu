	//assembly No 2
	//in this project order of variables in matrix State 
	//(in terminogy of original Python project)have been changed
	//on the following below order:
	//1-st variable - number of row of State (it is quantity of submatrixes)
	//2-nd variable - number of frequency (it is quantity of rows of each of submatrix)
	//3-rd variable - number of T (quantuty of columns of input image)
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "FdmtCu2.cuh"
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

char strInpFolder[] = "..//FDMT_TESTS//1024";

char strPathOutImageNpyFile_gpu[] = "out_image_GPU.npy";
const bool BDIM_512_1024 = true;


//extern int IROWS = 0;
//extern int ICOLS = 0;
//extern  std::vector<std::vector<int>> ivctOut = std::vector<std::vector<int>>(1, std::vector<int>(1, 0));


//--------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
	//// test init
	//int n = 4;
	//int* iarrin = (int*)malloc(n * n * sizeof(int));

	//int* d_iarrin = 0;
	//cudaMalloc(&d_iarrin, n * n * sizeof(int));

	//for (int i = 0; i < n * n; ++i)
	//{
	//	iarrin[i] = i;
	//}
	//cudaMemcpy(d_iarrin, iarrin, n * n * sizeof(int), cudaMemcpyHostToDevice);
	//
	//int* d_out =0;
	//cudaMalloc(&d_out, n * n * 100*sizeof(int));
	//int IDeltaT0 = 2;


	//fnc_init_fdmt(d_iarrin, n, n, IDeltaT0, d_out);


	//int *out =  (int*)malloc(n * n * 100*sizeof(int));
	//cudaMemcpy(out, d_out, n * n * (IDeltaT0 +1)*sizeof(int), cudaMemcpyDeviceToHost);
	//int in = 0;
	//for (int i = 0; i < (IDeltaT0 + 1); ++i)
	//{
	//	for (int j = 0; j < n; ++j)
	//	{
	//		for (int k = 0; k < n; ++k)

	//		{
	//			std::cout << out[in]<< ";";
	//			++in;
	//		}
	//		std::cout << std::endl;
	//	}
	//	std::cout << std::endl;
	//	std::cout << "------------------" << std::endl;

	//}
	//free(iarrin);
	//free(out);
	//
	//cudaFree(d_iarrin);
	//cudaFree(d_out);
	//int uuy = 0;
	//---------------------------------------------------------


	/*const int IDim0 = 5;
	const int IDim1 = 100;
	const int IDim2 = 200;*/

	/*const int IDim0 = 5;
	const int IDim1 = 256;
	const int IDim2 = 512;*/

	/*const int IDim0 = 4;
	const int IDim1 = 2;
	const int IDim2 = 1024;*/


	/*int* iarr = (int*)malloc(IDim0 * IDim1 * IDim2 * sizeof(int));

	int* iarrout = (int*)malloc(IDim0 * IDim1 * IDim2 * sizeof(int));

	for (int i = 0; i < IDim0 * IDim1 * IDim2; ++i)
	{
		iarr[i] = 0;
	}

	int* d_iarr = 0;
	cudaMallocManaged(&d_iarr, IDim0 * IDim1 * IDim2 * sizeof(int));
	cudaMemcpy(d_iarr, iarr, IDim0 * IDim1 * IDim2 * sizeof(int), cudaMemcpyHostToDevice);
	const dim3 blockSize1 = dim3(1, 1, 64);

	const dim3 gridSize1 = dim3(IDim0, IDim1, (IDim2 + blockSize1.z - 1) / blockSize1.z);



	kernel_0<<< gridSize1, blockSize1>>>( d_iarr, IDim0, IDim1,  IDim2);

	cudaMemcpy(iarrout, d_iarr, IDim0 * IDim1 * IDim2 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for (int i = 0; i < IDim0 * IDim1 * IDim2; ++i)
	{
		std::cout <<"arr["<< i << "]" << "=  " << iarrout[i] << std::endl;
	}

	free(iarr);
	free(iarrout);
	cudaFree(d_iarr);

	int iend = 0;*/
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
// 1. initiate pointer to input image
	int* piarr = (int*)malloc(sizeof(int));
	// !1

		// 2. initiate 2-pointer to input image, in order to realloc memory to satisfy arbitrary dimensions
	int** ppiarrImage = &piarr;
	// !2

	// 3. initiating input variables
	int iMaxDT = 0;
	int iImRows = 0, iImCols = 0;
	float val_fmax = 0., val_fmin = 0.;
	// !3

	// 4. reading input files from folder 
	int ireturn = downloadInputData(strInpFolder, &iMaxDT, ppiarrImage, &iImRows, &iImCols,
		&val_fmin, &val_fmax);
	// !4

	// 5. analysis output of reading function
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
	// !5

	// 5.1
	if ((iImRows == 1024) && (BDIM_512_1024))
	{
		iImRows = 512;
		iMaxDT = 512;
		piarr = (int*)realloc(piarr, iImRows * iImCols * sizeof(int));

	}
	// ! 5.1

	// 6. declare constants
	const int IMaxDT = iMaxDT;
	const int IImgrows = iImRows;
	const int IImgcols = iImCols;
	const float VAlFmin = val_fmin;
	const float VAlFmax = val_fmax;
	// !6 

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


	// 2. quant iteration's calculation
	const int I_F = (int)(log2((double)(IImgrows)));
	// !2

	// 3. temp variables calculations
	const float VAl_dF = (VAlFmax - VAlFmin) / ((float)(IImgrows));
	const int IDeltaT = int(ceil((IMaxDT - 1.) * (1. / (VAlFmin * VAlFmin)
		- 1. / ((VAlFmin + VAl_dF) * (VAlFmin + VAl_dF)))
		/ (1. / (VAlFmin
			* VAlFmin) - 1. / (VAlFmax * VAlFmax))));
	// !3


	// 4. declare pointers to device arrays	
	int* d_piarrState0 = 0;
	int* d_piarrState1 = 0;
	// !4

	// 5. allocate memory to device arrays	
	cudaMalloc(&d_piarrState0, IImgrows * (IDeltaT + 1) * IImgcols * sizeof(int));
	cudaMalloc(&d_piarrState1, IImgrows * (IDeltaT + 1) * IImgcols * sizeof(int));
	// !5

	// 6.  Initialize the device arrays with zeros	
	cudaMemset(d_piarrState0, 0, IImgrows * (IDeltaT + 1) * IImgcols * sizeof(int));
	cudaMemset(d_piarrState1, 0, IImgrows * (IDeltaT + 1) / 2 * IImgcols * sizeof(int));
	// !6

	// 7. allocate memory to device  auxiliary arrays	
	float* d_arr_val0 = 0;
	cudaMalloc(&d_arr_val0, IImgrows / 2 * sizeof(float));

	float* d_arr_val1 = 0;
	cudaMalloc(&d_arr_val1, IImgrows / 2 * sizeof(float));

	int* d_arr_deltaTLocal = 0;
	cudaMalloc(&d_arr_deltaTLocal, IImgrows / 2 * sizeof(int));

	int* d_arr_dT_MI = 0;
	//cudaMalloc(&d_arr_dT_MI, IImgrows * (IDeltaT + 1) * sizeof(int));

	int* d_arr_dT_ML = 0;
	//cudaMallocManaged(&d_arr_dT_ML, IImgrows * (IDeltaT + 1) * sizeof(int));

	int* d_arr_dT_RI = 0;
	//cudaMalloc(&d_arr_dT_RI, IImgrows * (IDeltaT + 1) * sizeof(int));
	// 7!


	// 8. calculations	
	int num = 1;
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < num; ++i)
	{
		fncFdmt_cu_v2(piarrImage  // input image
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
	
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	
	std::cout << "Time taken by function fncFdmt_cu_v0: " << duration.count() / ((double)num) << " milliseconds" << std::endl;
	// !8
	
	// 9. output in .npy:IImgcols * IMaxDT * sizeof(int));
	int * piarrImOut = (int*)malloc(IImgcols * IMaxDT * sizeof(int));
	cudaMemcpy(piarrImOut, u_piarrImOut, IImgcols* IMaxDT * sizeof(int), cudaMemcpyDeviceToHost);
	std::vector<int> v1(piarrImOut, piarrImOut + IImgcols * IMaxDT);

	std::array<long unsigned, 2> leshape101 {IImgcols , IMaxDT};

	npy::SaveArrayAsNumpy(strPathOutImageNpyFile_gpu, false, leshape101.size(), leshape101.data(), v1);
	// !9
	
	
	//--------------------------------------------------------------------------------------------------------------
	//-------------------- end of calculations ------------------------------------------------------------------------------------------
	//-------------------  drawing of output image for cuda -------------------------------------------------------------------------------------------
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
	cout << "GFLP/sec = " << ((double)flops) / ((double)duration.count() / 10.) * 1000. << "  GFP" << endl;
	
	free(piarrImOut);
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
	cudaFree(u_piarrImOut);
	
	

	char filename_cpu[] = "image_cpu.png";

	createImg_(argc, argv, v1, IImgcols, IMaxDT, filename_cpu);



	return 0;
}
