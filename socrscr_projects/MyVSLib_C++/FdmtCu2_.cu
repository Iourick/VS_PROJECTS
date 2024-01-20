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
#include "FdmtCu2_.cuh"
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
void fncFdmt_cu_v2_(int* piarrImage // input image
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
	, int* d_arr_dT_MI			// on-device auxiliary memory buffer = NULL, not in use
	, int* d_arr_dT_ML			// on-device auxiliary memory buffer = NULL, not in use
	, int* d_arr_dT_RI			// on-device auxiliary memory buffer = NULL, not in use
	, const  float VAlFmin
	, const  float VAlFmax
	, const int IMaxDT
	, int* u_piarrImOut			// OUTPUT image, dim = IDeltaT x IImgcols
)
{
	// 1. copying input rastr from Host to Device
	cudaMemcpy(d_piarrImage, piarrImage, IImgcols * IImgrows * sizeof(int), cudaMemcpyHostToDevice);
	// !1

	// 2. call initialization func
	clock_t start = clock();

	fnc_init_fdmt_v2_(d_piarrImage, IImgrows, IImgcols, IDeltaT, d_piarrState0);
	clock_t end = clock();
	double duration = double(end - start) / CLOCKS_PER_SEC;
	std::cout << "Time taken by fnc_init: " << duration << " seconds" << std::endl;
	// !2


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
	start = clock();
	for (int iit = 1; iit < (I_F + 1); ++iit)
	{
		fncFdmtIteration_v2_(d_p0, VAl_dF, iInp0, iInp1
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
	end = clock();
	duration = double(end - start) / CLOCKS_PER_SEC;
	std::cout << "Time taken by iterations: " << duration << " seconds" << std::endl;
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
// d_piarrInp �����  ����������� IDim0, IDim1,IDim2
// IDim1: this is iImgrows - quantity of rows of input power image, this is F
// IDim0: changes 
// IDim2: this is iImgcols - quantity of cols of input power image, this is T 
void fncFdmtIteration_v2_(int* d_piarrInp, const float val_dF, const int IDim0, const int IDim1
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

	/*create_auxillary_1d_arrays << <numberOfBlocks, threadsPerBlock >> > (iOutPutDim1
		, IMaxDT, VAlTemp1, VAlC2, VAlFmin, val_correction
		, d_arr_val0, d_arr_val1, d_iarr_deltaTLocal);
	cudaDeviceSynchronize();*/
	// !4

	// 
	//------------  KERNEL 3D kernel3D_shift_and_sum_v21  -----------------------------------
	//-----------  COALESCED + NOT SHARED MEMORY +  calc3AuxillaryVars ---------------------------------------------------------------------- 
	//----------- TIME = 104 /14 ms ----------------------------------	
	/*const dim3 blockSize1 = dim3(64, 1, 1);
	const dim3 gridSize1 = dim3((IDim2 + blockSize1.x - 1) / blockSize1.x, iOutPutDim1, iOutPutDim0);
	size_t smemsize = 32;
	kernel3D_shift_and_sum_v21 << < gridSize1, blockSize1, smemsize >> > (d_piarrInp
		, IDim0, IDim1, IDim2, d_iarr_deltaTLocal, d_arr_val0, d_arr_val1
		, iOutPutDim0, iOutPutDim1, d_piarrOut);
	cudaDeviceSynchronize();*/

	//------------  KERNEL 3D kernel3D_shift_and_sum_v2  -----------------------------------
	//-----------  COALESCED + SHARED MEMORY +  calc3AuxillaryVars ---------------------------------------------------------------------- 
	//----------- TIME = 100 /14 ms ----------------------------------	
	const dim3 blockSize1 = dim3(64, 1, 1);
	const dim3 gridSize1 = dim3((IDim2 + blockSize1.x - 1) / blockSize1.x, iOutPutDim1, iOutPutDim0);
	size_t smemsize = 32;
	kernel3D_shift_and_sum_v2_ << < gridSize1, blockSize1, smemsize >> > (d_piarrInp
		, IDim0, IDim1, IDim2, d_iarr_deltaTLocal, d_arr_val0, d_arr_val1
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
//IDim0 - quantity of submatrixes, = iDeltaT +1
//IDim1 - quantity of F, or quantity of rows of submatrix, "output_dims[0] = output_dims[0]//2;"
//IDim2 - quantity of cols of submattrix, IDim2 = quant cols of input image	
// d_iarr_deltaTLocal is one dimensional array with with length = IOutPutDim1
// d_iarr_dT_MI, d_iarr_dT_ML, d_iarr_dT_RI - are 2 dimensional arrays with dimension:IOutPutDim1 x IDim2
// 
__global__
void kernel3D_shift_and_sum_v2_(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, float* d_arr_val0, float* d_arr_val1
	, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut)
{
	__shared__ int shared_iarr[8];

	int i_F = blockIdx.y * blockDim.y + threadIdx.y;
	int i_dT = blockIdx.z * blockDim.z + threadIdx.z;
	shared_iarr[0] = i_dT;
	shared_iarr[1] = d_iarr_deltaTLocal[i_F];
	shared_iarr[5] = i_F;
	shared_iarr[6] = IOutPutDim1 * IDim2;
	shared_iarr[7] = IDim1 * IDim2;
	calc3AuxillaryVars_(d_iarr_deltaTLocal[i_F], i_dT, i_F, d_arr_val0[i_F]
		, d_arr_val1[i_F], shared_iarr[2], shared_iarr[4], shared_iarr[3]);
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
		d_piarrOut[numElem] = d_piarrInp[numInpElem0];// + d_piarrInp[numInpElem1 - shared_iarr[4]];
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
void kernel3D_shift_and_sum_v21_(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, float* d_arr_val0, float* d_arr_val1
	, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut)
{
	int numCol = blockIdx.x * blockDim.x + threadIdx.x;
	int i_F = blockIdx.y * blockDim.y + threadIdx.y;
	int i_dT = blockIdx.z * blockDim.z + threadIdx.z;
	if ((i_dT > d_iarr_deltaTLocal[i_F])||(i_F >= IDim1)||(numCol >= IDim2))
	{
		return;
	}
	int idT_middle_index = 0, idT_middle_larger = 0, idT_rest_index = 0;
	calc3AuxillaryVars_(d_iarr_deltaTLocal[i_F], i_dT, i_F, d_arr_val0[i_F]
		, d_arr_val1[i_F], idT_middle_index, idT_middle_larger, idT_rest_index);
	int numElem = i_dT * IOutPutDim1 * IDim2 + i_F * IDim2 + numCol;

	int numInpElem0 = idT_middle_index * IDim1 * IDim2 + 2 * i_F * IDim2 + numCol;

	int numInpElem1 = idT_rest_index * IDim1 * IDim2 + (1 + 2 * i_F) * IDim2 + numCol;
	// 	
	if (numCol >= idT_middle_larger)

	{
		//d_piarrOut[numElem] = d_piarrInp[numInpElem0] + d_piarrInp[numInpElem1 - idT_middle_larger];
	}
	else
	{
		d_piarrOut[numElem] = d_piarrInp[numInpElem0];
	}
}
//--------------------------------------------------------------------------
__host__ __device__
void calc3AuxillaryVars_(int& ideltaTLocal,int & i_dT, int& iF, float & val0
	, float& val1,int& idT_middle_index, int& idT_middle_larger, int& idT_rest_index)
{
	if (i_dT > ideltaTLocal)
	{
		idT_middle_index = 0;
		idT_middle_larger = 0;
		idT_rest_index = 0;
		return;
	}

	idT_middle_index = round(((float)i_dT) * val0);
	int ivalt = round(((float)i_dT) * val1);
	idT_middle_larger = ivalt;
	idT_rest_index = i_dT - ivalt;
}


//
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

//--------------------------------------------------------------------------------------
void fnc_init_fdmt_v2_(int* d_piarrImg, const int IImgrows, const int IImgcols
	, const int IDeltaT, int* d_piarrOut)
{
	//// output in .npy:
	//	int* parr = (int*)malloc(IImgrows * IImgcols  * sizeof(int));
	//	cudaMemcpy(parr, d_piarrImg, IImgrows * IImgcols  * sizeof(int)
	//		, cudaMemcpyDeviceToHost);
	//	std::vector<int> v6(parr, parr + IImgrows * IImgcols );

	//	std::array<long unsigned, 2> leshape126 {IImgrows,IImgcols};

	//	npy::SaveArrayAsNumpy("init00.npy", false, leshape126.size(), leshape126.data(), v6);
	//	free(parr);
	cudaMemcpy(d_piarrOut, d_piarrImg, IImgrows * IImgcols * sizeof(int)
		, cudaMemcpyDeviceToDevice);

	//int threadsPerBlock = 512;
	//const dim3 gridSize = dim3(IImgrows, (IImgcols + threadsPerBlock - 1) / threadsPerBlock);


	const dim3 blockSize = dim3(1, 256);
	const dim3 gridSize = dim3((IImgrows + blockSize.x - 1) / blockSize.x, (IImgcols + blockSize.y - 1) / blockSize.y);

	for (int i_dT = 1; i_dT < (IDeltaT + 1); ++i_dT)
	{

		kernel_init_iter_v2_<< < gridSize, blockSize >> > (d_piarrImg, IImgrows, IImgcols
			, i_dT, &d_piarrOut[(i_dT - 1) * IImgrows * IImgcols], &d_piarrOut[i_dT * IImgrows * IImgcols]);
		cudaDeviceSynchronize();
	}
}
//---------------------------------------------------------------------------------------------------

__global__
void kernel_init_iter_v2_(int* d_piarrImgRow, const int IImgrows, const int IImgcols
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
		//d_pMtrxCur[numElem] = d_pMtrxPrev[numElem] + d_piarrImgRow[iF * IImgcols + numElemInRow - i_dT];


	}
	else
	{
		d_pMtrxCur[numElem] = 0;
	}

}



