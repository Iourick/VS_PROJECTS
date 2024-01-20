//assembly No 1
	//in this project order of variables in matrix State 
	//(in terminogy of original Python project)have been changed
	//on the following below order:
	//1-st variable - number of row of each submatrix State (it is quantity of submatrixes)
	//2-nd variable - number of frequency (it is quantity of rows of each of submatrix)
	//3-rd variable - number of T (quantuty of columns of input image)
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------

#include "FdmtCu1.cuh"
#include <math.h>
#include <stdio.h>
#include <array>
#include <iostream>
#include <string>
#include <vector>
#include <ctime>   
#include "kernel.cuh"
#include <chrono>
#include "npy.hpp"


using namespace std;

//-------------------------------------------------------------------------

void fncFdmt_cu_v1(int* piarrImage // input image
	, int* d_piarrImage       // on-device auxiliary memory buffer
	, const int IImgrows, const int IImgcols // dimensions of input image 	
	, int* d_piarrState0		// on-device auxiliary memory buffer
	, int* d_piarrState1		// on-device auxiliary memory buffer
	, const int IDeltaT
	, const int I_F
	, const float VAl_dF
	, float* d_arr_val0 		// on-device auxiliary memory buffer
	, float* d_arr_val1			// on-device auxiliary memory buffer
	, int* d_arr_deltaTLocal	// on-device auxiliary memory buffer
	, int* d_arr_dT_MI			// on-device auxiliary memory buffer
	, int* d_arr_dT_ML			// on-device auxiliary memory buffer
	, int* d_arr_dT_RI			// on-device auxiliary memory buffer
	, const  float VAlFmin
	, const  float VAlFmax
	, const int IMaxDT
	, int* u_piarrImOut			// OUTPUT image, dim = IDeltaT x IImgcols
)
{
	// 1. copying input rastr from Host to Device
	cudaMemcpy(d_piarrImage, piarrImage, IImgcols * IImgrows * sizeof(int), cudaMemcpyHostToDevice);
	// !1

	
	
	 // 2. call initialization func, 560 microsec 2048x2048
	auto start2 = std::chrono::high_resolution_clock::now();

	const dim3 blockSize = dim3(1024, 1);
	const dim3 gridSize = dim3((IImgcols + blockSize.x - 1) / blockSize.x, (IImgrows + blockSize.y - 1) / blockSize.y);
	kernel_init_yk0 << < gridSize, blockSize >> > (d_piarrImage, IImgrows, IImgcols
		, IDeltaT, d_piarrState0);
	cudaDeviceSynchronize();	
	
	auto end2 = std::chrono::high_resolution_clock::now();
	auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
	std::cout << "Time taken by function fnc_init_fdmt_v12: " << duration2.count() /100. << " microseconds" << std::endl;
	// ! 2

	// 3.pointers initialization
	int* d_p0 = d_piarrState0;
	int* d_p1 = d_piarrState1;
	// 3!

	// 4. calculation dimensions of input State matrix for iteration process
	int iInp1 = IImgrows;
	int iInp0 = IDeltaT + 1;
	// !4

	// 5. declare variables to keep dimansions of output state 
	int iOut0 = 0, iOut1 = 0, iOut2 = 0;
	// !5

	// 7. iterations
	auto start1 = clock();
	for (int iit = 1; iit < (I_F + 1); ++iit)
	{
		fncFdmtIteration_v1(d_p0, VAl_dF, iInp0, iInp1
			, IImgcols, IMaxDT, VAlFmin
			, VAlFmax, iit, d_arr_val0
			, d_arr_val1, d_arr_deltaTLocal
			, d_arr_dT_MI, d_arr_dT_ML, d_arr_dT_RI
			, d_p1, iOut0, iOut1);
		if (iit == I_F)
		{
			break;
		}
		// exchange order of pointers
		int* d_pt = d_p0;
		d_p0 = d_p1;
		d_p1 = d_pt;
		iInp0 = iOut0;
		iInp1 = iOut1;
		if (iit == I_F - 1)
		{
			d_p1 = u_piarrImOut;
		}
		// !
	}
	auto end1 = clock();
	double duration1 = double(end1 - start1) / CLOCKS_PER_SEC;
	std::cout << "Time taken by iterations: " << duration1 << " seconds" << std::endl;
	// ! 7


}

//--------------------------------------------------------------------------------------
//    Input :
//    Input - 3d array, with dimensions[N_d0,N_f,  Nt]
//    f_min, f_max - are the base - band begin and end frequencies.
//    The frequencies can be entered in both MHz and GHz, units are factored out in all uses.
//    maxDT - the maximal delay(in time bins) of the maximal dispersion.
//    Appears in the paper as N_{\Delta}
//A typical input is maxDT = N_f
//dataType - To naively use FFT, one must use floating point types.
//Due to casting, use either complex64 or complex128.
//iteration num - Algorithm works in log2(Nf) iterations, each iteration changes all the sizes(like in FFT)
//Output:
//3d array, with dimensions[N_d1,N_f / 2,  Nt]
//    where N_d1 is the maximal number of bins the dispersion curve travels at one output frequency band
//
//    For details, see algorithm 1 in Zackay & Ofek(2014)
// F,T = Image.shape 
// d_piarrInp имеет  размерности IDim0, IDim1,IDim2
// IDim1: this is iImgrows - quantity of rows of input power image, this is F
// IDim0: changes 
// IDim2: this is iImgcols - quantity of cols of input power image, this is T 
void fncFdmtIteration_v1(int* d_piarrInp, const float val_dF, const int IDim0, const int IDim1
	, const int IDim2, const int IMaxDT, const float VAlFmin
	, const float VAlFmax, const int ITerNum, float* d_arr_val0
	, float* d_arr_val1, int* d_iarr_deltaTLocal
	, int* d_iarr_dT_MI, int* d_iarr_dT_ML, int* d_iarr_dT_RI
	, int* d_piarrOut, int& iOutPutDim0, int& iOutPutDim1)
{
	// 1. calculation of dimensions of output Stae mtrx(=d_piarrOut)
	float valDeltaF = pow(2., ITerNum) * val_dF;
	float temp0 = 1. / (VAlFmin * VAlFmin) -
		1. / ((VAlFmin + valDeltaF) * (VAlFmin + valDeltaF));

	const float VAlTemp1 = 1. / (VAlFmin * VAlFmin) - 1. / (VAlFmax * VAlFmax);

	int ideltaT = (int)(ceil(((float)IMaxDT - 1.0) * temp0 / VAlTemp1));
	iOutPutDim0 = ideltaT + 1;
	iOutPutDim1 = IDim1 / 2;
	// !1


	// 2. set zeros in output array
	cudaMemset(d_piarrOut, 0, iOutPutDim0 * iOutPutDim1 * IDim2 * sizeof(int));
	// !2

	// 3. constants calculation
	float val_correction = 0;
	if (ITerNum > 0)
	{
		val_correction = val_dF / 2.;
	}
	const float VAlC2 = (VAlFmax - VAlFmin) / ((float)(iOutPutDim1));
	// !3	

	// 4. calculating first 3 auxillary 1 dim arrays
	int threadsPerBlock = 1024;
	int numberOfBlocks = (iOutPutDim1 + threadsPerBlock - 1) / threadsPerBlock;

	create_auxillary_1d_arrays << <numberOfBlocks, threadsPerBlock >> > (iOutPutDim1
		, IMaxDT, VAlTemp1, VAlC2, VAlFmin, val_correction
		, d_arr_val0, d_arr_val1, d_iarr_deltaTLocal);
	cudaDeviceSynchronize();
	// !4

	// 5. calculating second 3 auxillary 2 dim arrays
	
	auto start = std::chrono::high_resolution_clock::now();
	const dim3 blockSize = dim3(1, 1024);
	const dim3 gridSize = dim3((iOutPutDim1 + blockSize.x - 1) / blockSize.x, (iOutPutDim0 + blockSize.y - 1) / blockSize.y);
	kernel_create_aux_2d_arrays_v1 << < gridSize, blockSize >> > (iOutPutDim1
		, iOutPutDim0, d_arr_val0, d_arr_val1, d_iarr_deltaTLocal
		, d_iarr_dT_MI, d_iarr_dT_ML, d_iarr_dT_RI);
	cudaDeviceSynchronize();
	
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "Time taken by kernel_2d_arrays_v1: " << duration.count() << " microseconds" << std::endl;
	
	// !5
//---------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------
//----------   6. Iteration's performing   -----------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------
	//
	//// --------- KERNEL 2D kernel2D_shift_and_sum_v1 --------------------------------------
	////----------- TIME = 110 / 17 ms ----------------------------------
	/*const dim3 blockSize1 = dim3(64, 1);
	const dim3 gridSize1 = dim3( (IDim2 + blockSize1.x - 1) / blockSize1.x,iOutPutDim1* iOutPutDim0);
	kernel2D_shift_and_sum_v1<<< gridSize1, blockSize1>>>(d_piarrInp, IDim0, IDim1
		, IDim2, d_iarr_deltaTLocal, d_iarr_dT_MI
		, d_iarr_dT_ML, d_iarr_dT_RI, iOutPutDim0, iOutPutDim1
		, d_piarrOut);*/

	
	////------------  KERNEL 1D V1-----------------------------------
	////----------- TIME = 150 / 19 ms ----------------------------------
	/*threadsPerBlock = 256;
	const int quantBlocksPerRow = (IDim2 + threadsPerBlock - 1) / threadsPerBlock;
	numberOfBlocks = iOutPutDim0 * iOutPutDim1 * quantBlocksPerRow;
	kernel1D_shift_and_sum_v1<<< numberOfBlocks, threadsPerBlock>>>(quantBlocksPerRow
		, d_piarrInp,  IDim0,  IDim1, IDim2,  d_iarr_deltaTLocal, d_iarr_dT_MI
		,  d_iarr_dT_ML, d_iarr_dT_RI, iOutPutDim0, iOutPutDim1
		, d_piarrOut);
	cudaDeviceSynchronize();*/


	//------------  KERNEL 1D kernel_shift_and_sum_  -----------------------------------
	//----------- TIME = 134 / 19 ms ----------------------------------
	/*threadsPerBlock = 256;
	numberOfBlocks = (iOutPutDim0 * iOutPutDim1 * IDim2 + threadsPerBlock - 1) / threadsPerBlock;	
	kernel_shift_and_sum_ << < numberOfBlocks, threadsPerBlock >> > (d_piarrInp, IDim0, IDim1
		, IDim2,  d_iarr_deltaTLocal,  d_iarr_dT_MI
		, d_iarr_dT_ML,  d_iarr_dT_RI, iOutPutDim0,  iOutPutDim1
		,  d_piarrOut);*/


	//------------  KERNEL 1D kernel1D_shift_and_sum_v11  -----------------------------------
	//----------- TIME = 170/18 ms ----------------------------------	
	//threadsPerBlock = 512;	
	//const int quantBlocksPerRow = (IDim2 + threadsPerBlock - 1) / threadsPerBlock;
	//numberOfBlocks = iOutPutDim0 * iOutPutDim1 * quantBlocksPerRow;
	//kernel1D_shift_and_sum_v11 << < numberOfBlocks, threadsPerBlock >> > ( quantBlocksPerRow,  d_piarrInp,  IDim0,  IDim1
	//	, IDim2, d_iarr_deltaTLocal, d_arr_val0, d_arr_val1/*, int* d_iarr_dT_MI
	//	, int* d_iarr_dT_ML, int* d_iarr_dT_RI*/, iOutPutDim0, iOutPutDim1
	//	, d_piarrOut);
	//cudaDeviceSynchronize();




	//------------  KERNEL 3D kernel3D_shift_and_sum_v11  -----------------------------------
	//-----------  NOT COALESCED   ---------------------------------------------------------------------- 
	//----------- TIME = 86 /17 ms ----------------------------------	
	/*const dim3 blockSize1 = dim3(1, 1, 64);
	const dim3 gridSize1 = dim3(iOutPutDim0, iOutPutDim1,  (IDim2 + blockSize1.z - 1) / blockSize1.z);
	size_t smemsize = 32;
	kernel3D_shift_and_sum_v11<<< gridSize1, blockSize1, smemsize >>>(d_piarrInp
		, IDim0, IDim1, IDim2, d_iarr_deltaTLocal, d_iarr_dT_MI, d_iarr_dT_ML, d_iarr_dT_RI
		, iOutPutDim0, iOutPutDim1, d_piarrOut);
	cudaDeviceSynchronize();*/


	//------------  kernel3D_shift_and_sum_v12  -----------------------------------
	//-----------  COALESCED   ---------------------------------------------------------------------- 
	//----------- TIME = 84/15 ms ----------------------------------	
	const dim3 blockSize1 = dim3(64, 1, 1);
	const dim3 gridSize1 = dim3( (IDim2 + blockSize1.x - 1) / blockSize1.x,iOutPutDim1, iOutPutDim0);
	size_t smemsize = 32;
	kernel3D_shift_and_sum_v12 << < gridSize1, blockSize1, smemsize >> > (d_piarrInp
		, IDim0, IDim1, IDim2, d_iarr_deltaTLocal, d_iarr_dT_MI, d_iarr_dT_ML, d_iarr_dT_RI
		, iOutPutDim0, iOutPutDim1, d_piarrOut);
	cudaDeviceSynchronize();
		

	/*int* parr1 = (int*)malloc(iOutPutDim0 * iOutPutDim1 * IDim2  * sizeof(int));
	cudaMemcpy(parr1, d_piarrOut, iOutPutDim0 * iOutPutDim1 * IDim2 * sizeof(int)
		, cudaMemcpyDeviceToHost);
	std::vector<int> v7(parr1, parr1 + iOutPutDim0 * iOutPutDim1 * IDim2);

	std::array<long unsigned, 3> leshape127 {iOutPutDim0, iOutPutDim1, IDim2};

	npy::SaveArrayAsNumpy("new.npy", false, leshape127.size(), leshape127.data(), v7);
	free(parr1);
	int ii = 0;*/

}
//-----------------------------------------------------------------------------------------------------------------------
// BAD DIFFERENT WAY OF INDEXING ORDER IN THE KERNEL, NOT COALECSED
//IDim0 - quantity of submatrixes, = iDeltaT +1
//IDim1 - quantity of F, or quantity of rows of submatrix, "output_dims[0] = output_dims[0]//2;"
//IDim2 - quantity of cols of submattrix, IDim2 = quant cols of input image	
// d_iarr_deltaTLocal is one dimensional array with with length = IOutPutDim1
// d_iarr_dT_MI, d_iarr_dT_ML, d_iarr_dT_RI - are 2 dimensional arrays with dimension:IOutPutDim1 x IDim2 
__global__
void kernel3D_shift_and_sum_v11(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut)
{
	__shared__ int shared_iarr[8];

	int i_dT = blockIdx.x * blockDim.x + threadIdx.x;
	int i_F = blockIdx.y * blockDim.y + threadIdx.y;
	shared_iarr[0] = i_dT;
	shared_iarr[1] = d_iarr_deltaTLocal[i_F];
	int itemp = i_F * IOutPutDim0 + i_dT;
	shared_iarr[2] = d_iarr_dT_MI[itemp];
	shared_iarr[3] = d_iarr_dT_RI[itemp];
	shared_iarr[4] = d_iarr_dT_ML[itemp];
	shared_iarr[5] = i_F;
	shared_iarr[6] = IOutPutDim1 * IDim2;
	shared_iarr[7] = IDim1 * IDim2;
	__syncthreads();

	if (shared_iarr[0] > shared_iarr[1])
	{
		return;
	}
	int numCol = blockIdx.z * blockDim.z + threadIdx.z;
	if (numCol > IDim2)
	{
		return;
	}

	int numElem = shared_iarr[0] * shared_iarr[6] + shared_iarr[5] * IDim2 + numCol;

	int numInpElem0 = shared_iarr[2] * shared_iarr[7] + 2 * shared_iarr[5] * IDim2 + numCol;

	int numInpElem1 = shared_iarr[3] * shared_iarr[7] + (1 + 2 * shared_iarr[5]) * IDim2 + numCol;
	// 	
	if (numCol >= shared_iarr[4])

	{
		d_piarrOut[numElem] = d_piarrInp[numInpElem0] + d_piarrInp[numInpElem1 - shared_iarr[4]];
	}
	else
	{
		d_piarrOut[numElem] = d_piarrInp[numInpElem0];
	}

}
//-----------------------------------------------------------------------------------------------------------------------
//DIFFERENT WAY OF INDEXING ORDER IN THE KERNEL COALECSED
//IDim0 - quantity of submatrixes, = iDeltaT +1
//IDim1 - quantity of F, or quantity of rows of submatrix, "output_dims[0] = output_dims[0]//2;"
//IDim2 - quantity of cols of submattrix, IDim2 = quant cols of input image	
// d_iarr_deltaTLocal is one dimensional array with with length = IOutPutDim1
// d_iarr_dT_MI, d_iarr_dT_ML, d_iarr_dT_RI - are 2 dimensional arrays with dimension:IOutPutDim1 x IDim2
// 
__global__
void kernel3D_shift_and_sum_v12(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut)
{
	__shared__ int shared_iarr[8];

	int i_F = blockIdx.y * blockDim.y + threadIdx.y;
	int i_dT = blockIdx.z * blockDim.z + threadIdx.z;
	shared_iarr[0] = i_dT;
	shared_iarr[1] = d_iarr_deltaTLocal[i_F];
	int itemp = i_F * IOutPutDim0 + i_dT;
	shared_iarr[2] = d_iarr_dT_MI[itemp];
	shared_iarr[3] = d_iarr_dT_RI[itemp];
	shared_iarr[4] = d_iarr_dT_ML[itemp];
	shared_iarr[5] = i_F;
	shared_iarr[6] = IOutPutDim1 * IDim2;
	shared_iarr[7] = IDim1 * IDim2;
	__syncthreads();

	if (shared_iarr[0] > shared_iarr[1])
	{
		return;
	}
	int numCol = blockIdx.x * blockDim.x + threadIdx.x;
	if (numCol > IDim2)
	{
		return;
	}

	int numElem = shared_iarr[0] * shared_iarr[6] + shared_iarr[5] * IDim2 + numCol;

	int numInpElem0 = shared_iarr[2] * shared_iarr[7] + 2 * shared_iarr[5] * IDim2 + numCol;

	int numInpElem1 = shared_iarr[3] * shared_iarr[7] + (1 + 2 * shared_iarr[5]) * IDim2 + numCol;
	// 	
	if (numCol >= shared_iarr[4])

	{
		d_piarrOut[numElem] = d_piarrInp[numInpElem0] + d_piarrInp[numInpElem1 - shared_iarr[4]];
	}
	else
	{
		d_piarrOut[numElem] = d_piarrInp[numInpElem0];
	}

}

//-----------------------------------------------------------------------------------------------------------------------
__global__
void kernel_shift_and_sum_(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= IOutPutDim0 * IOutPutDim1 * IDim2)
	{
		return;
	}
	int iw = IOutPutDim1 * IDim2;
	int i_dT = i / iw;
	int irest = i % iw;
	int i_F = irest / IDim2;
	if (i_dT > d_iarr_deltaTLocal[i_F])
	{
		return;
	}
	int idx = irest % IDim2;
	
	int ind = i_F * IOutPutDim0 + i_dT;
	
	d_piarrOut[i] = d_piarrInp[ d_iarr_dT_MI[ind] * IDim1 * IDim2 +
		2 * i_F * IDim2 + idx];	

	if (idx >= d_iarr_dT_ML[ind])
	{
		
		int indInpMtrx = d_iarr_dT_RI[ind] * IDim1 * IDim2 + (2 * i_F + 1) * IDim2 + idx - d_iarr_dT_ML[ind];
		
		d_piarrOut[i] += d_piarrInp[indInpMtrx];
	}
}
//-----------------------------------------------------------------------------------------------------------------
__global__
void kernel1D_shift_and_sum_v1(const int quantBlocksPerRow, int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= IOutPutDim0 * IOutPutDim1 * IDim2)
	{
		return;
	}
	int iw = IOutPutDim1 * IDim2;
	int i_dT = i / iw;
	int irest = i % iw;
	int i_F = irest / IDim2;
	if (i_dT > d_iarr_deltaTLocal[i_F])
	{
		return;
	}
	int idx = irest % IDim2;
	// claculation of bound index: 
	// arr_dT_ML[i_F, i_dT]
	// index of arr_dT_ML
	// arr_dT_ML is matrix with IOutPutDim0 rows and IOutPutDim1 cols
	int ind = i_F * IOutPutDim0 + i_dT;
	// !

	// calculation of:
	// d_Output[i_F][i_dT][idx] = d_input[2 * i_F][arr_dT_MI[i_F, i_dT]][idx]
	  // calculation num row of submatix No_2 * i_F of d_piarrInp = arr_dT_MI[ind]
	d_piarrOut[i] =  d_piarrInp[d_iarr_dT_MI[ind] * IDim1 * IDim2 + 2 * i_F * IDim2 + idx];

	//d_piarrOut[i] = d_piarrInp[2 * i_F * IDim1 * IDim2 +
		//d_iarr_dT_MI[ind] * IDim2 + idx];

	if (idx >= d_iarr_dT_ML[ind])
	{
		int numRow = d_iarr_dT_RI[ind];
		//int indInpMtrx = (2 * i_F + 1) * IDim1 * IDim2 + numRow * IDim2 + idx - d_iarr_dT_ML[ind];
		int indInpMtrx = numRow * IDim1 * IDim2 + (2 * i_F + 1) *  IDim2 + idx - d_iarr_dT_ML[ind];
		//atomicAdd(&d_piarrOut[i], d_piarrInp[ind]);
		d_piarrOut[i] += d_piarrInp[indInpMtrx];
	}
}

//------------------------------------------------------------------------------------
//IDim0 - quantity of submatrixes, = iDeltaT +1
//IDim1 - quantity of F, or quantity of rows of submatrix, "output_dims[0] = output_dims[0]//2;"
//IDim2 - quantity of cols of submattrix, IDim2 = quant cols of input image	
// d_iarr_deltaTLocal is one dimensional array with with length = IOutPutDim1
// d_iarr_dT_MI, d_iarr_dT_ML, d_iarr_dT_RI - are 2 dimensional arrays with dimension:IOutPutDim1 x IDim2

 __global__
void kernel1D_shift_and_sum_v11(const int quantBlocksPerRow, int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, float* d_arr_val0, float* d_arr_val1/*, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI*/, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut)
{

	__shared__ float shared_arr[2];
	__shared__ int shared_iarr[6];

	// quant elements per submatrix
	int quantBlocksPerMtrx = quantBlocksPerRow * IDim1;

	//number of submatrix
	int i_dT = (blockIdx.x) / quantBlocksPerMtrx;


	int i_rest = (blockIdx.x) % quantBlocksPerMtrx;

	// current row of submatrix number  
	int i_F = i_rest / quantBlocksPerRow;

	// current number of block in the row
	int numBlockInRow = i_rest % quantBlocksPerRow;


	shared_arr[0] = d_arr_val0[i_F];
	shared_arr[1] = d_arr_val1[i_F];
	shared_iarr[0] = d_iarr_deltaTLocal[i_F];
	shared_iarr[1] = IDim1 * IDim2;
	shared_iarr[2] = IOutPutDim1 * IDim2;
	shared_iarr[3] = i_F;
	shared_iarr[4] = i_dT;
	shared_iarr[5] = numBlockInRow;
	__syncthreads();

	if (shared_iarr[4] > shared_iarr[0])
	{
		return;
	}

	// number of col of element in row of submatrix
	int numElemCol = shared_iarr[5] * blockDim.x + threadIdx.x;

	if (numElemCol >= IDim2)
	{
		return;
	}
	
	
	int idT_middle_index = round(((float)shared_iarr[4]) * shared_arr[0]);
	int idT_middle_larger = round(((float)shared_iarr[4]) * shared_arr[1]);
	int idT_rest_index = shared_iarr[4] - idT_middle_larger;
	

	int numElem = shared_iarr[4] * shared_iarr[2] + shared_iarr[3] * IDim2 + numElemCol;
	if (numElem >= IOutPutDim0 * IOutPutDim1 * IDim2)
	{
		return;
	}
	//d_piarrOut[numElem] = numElem;
	
	// claculation of bound index: 
	// arr_dT_ML[i_F, i_dT]
	// index of arr_dT_ML
	// arr_dT_ML is matrix with IOutPutDim0 rows and IOutPutDim1 cols
	// position elemens of d_iarr_dT_MI, d_iarr_dT_ML,d_iarr_dT_RI

	// !
	// calculation indexes of input array
	// d_input[2 * i_F][arr_dT_MI[i_F, i_dT]][idx]
	int numInpElem0 = idT_middle_index * shared_iarr[1] + 2 * shared_iarr[3] * IDim2 + numElemCol;
	
	
	
	if (numElemCol >= idT_middle_larger)
	{
		int numInpElem1 = idT_rest_index * shared_iarr[1] + (1 + 2 * shared_iarr[3]) * IDim2 + numElemCol - idT_middle_larger;

		d_piarrOut[numElem] = d_piarrInp[numInpElem0] + d_piarrInp[numInpElem1];
		//atomicAdd(&d_piarrOut[numElem], d_piarrInp[idT_rest_index * shared_iarr[1] + (1 + 2 * shared_iarr[3]) * IDim2 + numElemCol - idT_middle_larger]);
	}
	else
	{
		d_piarrOut[numElem] = d_piarrInp[numInpElem0];
	}

}

//-----------------------------------------------------------------------------------------------------------------------
//IDim0 - quantity of submatrixes, = iDeltaT +1
//IDim1 - quantity of F, or quantity of rows of submatrix, "output_dims[0] = output_dims[0]//2;"
//IDim2 - quantity of cols of submattrix, IDim2 = quant cols of input image	
// d_iarr_deltaTLocal is one dimensional array with with length = IOutPutDim1
// d_iarr_dT_MI, d_iarr_dT_ML, d_iarr_dT_RI - are 2 dimensional arrays with dimension:IOutPutDim1 x IDim2
// 
__global__
void kernel3D_shift_and_sum_v1(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut)
{
	int i_dT = blockIdx.x * blockDim.x + threadIdx.x;
	int i_F = blockIdx.y * blockDim.y + threadIdx.y;
	if (i_dT > d_iarr_deltaTLocal[i_F])
	{
		return;
	}
	int quantMtxElems = IOutPutDim1 * IDim2;
	int numCol = blockIdx.z * blockDim.z + threadIdx.z;
	if (numCol > IDim2)
	{
		return;
	}

	int numElem = i_dT * quantMtxElems + i_F * IDim2 + numCol;
	

	int quantInpMtxElems = IDim1 * IDim2;
	int idT_middle_index = d_iarr_dT_MI[i_F * IOutPutDim0 + i_dT];
	int numInpElem0 = idT_middle_index * quantInpMtxElems + 2 * i_F * IDim2 + numCol;
	int idT_rest_index = d_iarr_dT_RI[i_F * IOutPutDim0 + i_dT];
	int numInpElem1 = idT_rest_index * quantInpMtxElems + (1 + 2 * i_F) * IDim2 + numCol;
	// 


	int idT_middle_larger = d_iarr_dT_ML[i_F * IOutPutDim0 + i_dT];
	if (numCol >= idT_middle_larger)

	{
		d_piarrOut[numElem] = d_piarrInp[numInpElem0] + d_piarrInp[numInpElem1 - idT_middle_larger];
	}
	else
	{
		d_piarrOut[numElem] = d_piarrInp[numInpElem0];
	}

}
//------------------------------------------------------------------------------------
//IDim0 - quantity of submatrixes, = iDeltaT +1
//IDim1 - quantity of F, or quantity of rows of submatrix, "output_dims[0] = output_dims[0]//2;"
//IDim2 - quantity of cols of submattrix, IDim2 = quant cols of input image	
// d_iarr_deltaTLocal is one dimensional array with with length = IOutPutDim1
// d_iarr_dT_MI, d_iarr_dT_ML, d_iarr_dT_RI - are 2 dimensional arrays with dimension:IOutPutDim1 x IDim0
// 
__global__
void kernel2D_shift_and_sum_v1(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut)
{
	int i_dT = (blockIdx.y * blockDim.y + threadIdx.y)/ IOutPutDim1;
	int i_F = (blockIdx.y * blockDim.y + threadIdx.y) % IOutPutDim1;
	if (i_dT > d_iarr_deltaTLocal[i_F])
	{
		return;
	}
	int quantMtxElems = IOutPutDim1 * IDim2;
	int numCol = blockIdx.x * blockDim.x + threadIdx.x;
	if (numCol > IDim2)
	{
		return;
	}

	int numElem = i_dT * quantMtxElems + i_F * IDim2 + numCol;
	//d_piarrInp[numElem] = numElem;

	int quantInpMtxElems = IDim1 * IDim2;
	int idT_middle_index = d_iarr_dT_MI[i_F * IOutPutDim0 + i_dT];
	int numInpElem0 = idT_middle_index * quantInpMtxElems + 2 * i_F * IDim2 + numCol;
	int idT_rest_index = d_iarr_dT_RI[i_F * IOutPutDim0 + i_dT];
	int numInpElem1 = idT_rest_index * quantInpMtxElems + (1 + 2 * i_F) * IDim2 + numCol; 	

	int idT_middle_larger = d_iarr_dT_ML[i_F * IOutPutDim0 + i_dT];
	if (numCol >= idT_middle_larger)
	{
		d_piarrOut[numElem] = d_piarrInp[numInpElem0] + d_piarrInp[numInpElem1 - idT_middle_larger];
	}
	else
	{
		d_piarrOut[numElem] = d_piarrInp[numInpElem0];
	}	
}


//-----------------------------------------------------------------------------------------------------------------------

__global__
void create_auxillary_1d_arrays(const int IFjumps, const int IMaxDT, const float VAlTemp1
	, const float VAlc2, const float VAlf_min, const float VAlcorrection
	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > IFjumps)
	{
		return;
	}
	float valf_start = VAlc2 * i + VAlf_min;
	float valf_end = valf_start + VAlc2;
	float valf_middle_larger = VAlc2 / 2. + valf_start + VAlcorrection;
	float valf_middle = VAlc2 / 2. + valf_start - VAlcorrection;
	float temp0 = 1. / (valf_start * valf_start) - 1. / (valf_end * valf_end);

	d_arr_val0[i] = -(1. / (valf_middle * valf_middle) - 1. / (valf_start * valf_start)) / temp0;

	d_arr_val1[i] = -(1. / (valf_middle_larger * valf_middle_larger)
		- 1. / (valf_start * valf_start)) / temp0;

	d_iarr_deltaTLocal[i] = (int)(ceil((((float)(IMaxDT)) - 1.) * temp0 / VAlTemp1));
}
//--------------------------------------------------------------------------------------

__global__
void kernel_create_aux_2d_arrays_v1(const int IDim0, const int IDim1
	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal
	, int* d_iarr_dT_middle_index, int* d_iarr_dT_middle_larger
	, int* d_iarr_dT_rest_index)

{
	int iF = blockIdx.x * blockDim.x;
	int numElem = blockIdx.x * blockDim.x * IDim1 + blockIdx.y * blockDim.y + threadIdx.y;	
	int i_dT = blockIdx.y * blockDim.y + threadIdx.y;
	if (i_dT >= IDim1)
	{
		return;
	}
	
	if (i_dT > d_iarr_deltaTLocal[iF])
	{
		d_iarr_dT_middle_index[numElem] = 0;
		d_iarr_dT_middle_larger[numElem] = 0;
		d_iarr_dT_rest_index[numElem] = 0;
		return;
	}

	d_iarr_dT_middle_index[numElem] = round(((float)i_dT) * d_arr_val0[iF]);
	int ivalt = round(((float)i_dT) * d_arr_val1[iF]);
	d_iarr_dT_middle_larger[numElem] = ivalt;
	d_iarr_dT_rest_index[numElem] = i_dT - ivalt;

}
//--------------------------------------------------------------------------------------
void fnc_init_fdmt_v1(int* d_piarrImg, const int IImgrows, const int IImgcols
	, const int IDeltaT, int* d_piarrOut)
{

	cudaMemcpy(d_piarrOut, d_piarrImg, IImgrows * IImgcols * sizeof(int)
		, cudaMemcpyDeviceToDevice);

	const dim3 blockSize = dim3(1, 256);
	const dim3 gridSize = dim3((IImgrows + blockSize.x - 1) / blockSize.x, (IImgcols + blockSize.y - 1) / blockSize.y);

	for (int i_dT = 1; i_dT < (IDeltaT + 1); ++i_dT)
	{

		kernel_init_iter_v1 << < gridSize, blockSize >> > (d_piarrImg, IImgrows, IImgcols
			, i_dT, &d_piarrOut[(i_dT - 1) * IImgrows * IImgcols], &d_piarrOut[i_dT * IImgrows * IImgcols]);
		cudaDeviceSynchronize();
	}
}



//---------------------------------------------------------------------------------------------------
// order of keeping vars = 1-0-2
__global__
void kernel_init_yk0(int* d_piarrImg, const int IImgrows, const int IImgcols
	, const int IDeltaT, int* d_piarrState0)
{
	int i_F = blockIdx.y;
	int numElemInRow = blockIdx.x * blockDim.x + threadIdx.x;
	if (numElemInRow >= IImgcols)
	{
		return;
	}

	int iInpIndCur = i_F * IImgcols + numElemInRow;
	int iOutIndCur = iInpIndCur;
	d_piarrState0[iOutIndCur] = d_piarrImg[iInpIndCur];
	int numElemInSubMtrx = IImgrows * IImgcols;
	int itemp = d_piarrState0[iOutIndCur];
	for (int i_dT = 1; i_dT < (IDeltaT + 1); ++i_dT)
	{
		iOutIndCur += numElemInSubMtrx;

		if (i_dT <= numElemInRow)
		{
			d_piarrState0[iOutIndCur] = itemp + d_piarrImg[iInpIndCur - i_dT];
			itemp = d_piarrState0[iOutIndCur];
		}
		else
		{
			d_piarrState0[iOutIndCur] = 0;
		}

	}

}
//---------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------

__global__
void kernel_init_yk1(int* d_piarrImg, const int IImgrows, const int IImgcols
	, const int IDeltaT, int* d_piarrState0)
{
	int i_F = blockIdx.y;
	int numElemInRow = blockIdx.x * blockDim.x + threadIdx.x;
	if (numElemInRow >= IImgcols)
	{
		return;
	}

	int iInpIndCur = i_F * IImgcols + numElemInRow;
	int iOutIndCur = iInpIndCur;
	int* piOut = d_piarrState0 + iOutIndCur;
	int* piImg = d_piarrImg + iOutIndCur;
	//d_piarrState0[iOutIndCur] = d_piarrImg[iInpIndCur];
	int numElemInSubMtrx = IImgrows * IImgcols;
	*piOut = *piImg;
	int* pitemp = piOut;
	for (int i_dT = 1; i_dT < (IDeltaT + 1); ++i_dT)
	{
		piOut += numElemInSubMtrx;
		--piImg;

		if (i_dT <= numElemInRow)
		{
			*piOut = *pitemp + (*piImg);
			
			pitemp = piOut;
		}
		else
		{
			*piOut = 0;
		}

	}

}
//---------------------------------------------------------------------------------------------------

__global__
void kernel_init_iter_v1(int* d_piarrImgRow, const int IImgrows, const int IImgcols
	, const int i_dT, int* d_pMtrxPrev, int* d_pMtrxCur)
{
	int iF = blockIdx.x * blockDim.x;
	int numElemInRow = blockIdx.y * blockDim.y + threadIdx.y;
	if (numElemInRow >= IImgcols)
	{
		return;
	}
	int numElem = blockIdx.x * IImgcols + numElemInRow;
	//Output[:, i_dT, i_dT : ] = Output[:, i_dT - 1, i_dT : ] + Image[:, : -i_dT]

	if (i_dT <= numElemInRow)
	{
		d_pMtrxCur[numElem] = d_pMtrxPrev[numElem] + d_piarrImgRow[iF * IImgcols + numElemInRow - i_dT];

	}
	else
	{
		d_pMtrxCur[numElem] = 0;
	}
}



