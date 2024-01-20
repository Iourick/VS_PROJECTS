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

#include "Constants.h"



using namespace std;

char strInpFolder[] = "..//FDMT_TESTS//512";

char strPathOutImageNpyFile_gpu[] = "out_image_GPU.npy";
const bool BDIM_512_1024 = true;




//--------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
	
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
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 10; ++i)
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
	
	std::cout << "Time taken by function fncFdmt_cu_v0: " << duration.count()/10. << " milliseconds" << std::endl;
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
			flops = GFLPS_1024;
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
	
	




	return 0;
}
