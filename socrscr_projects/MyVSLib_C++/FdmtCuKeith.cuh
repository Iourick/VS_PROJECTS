#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void fncFdmt_cu_keith(int* piarrImage // input image
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
	, const  float VAlFmin, const  float VAlFmax, const int IMaxDT
	, int* u_piarrImOut			// OUTPUT image, dim = IDeltaT x IImgcols
);

__global__
void kernel_init_iter_keith(int* d_piarrImgRow, const int IImgrows, const int IImgcols
	, const int i_dT, int* d_pMtrxPrev, int* d_pMtrxCur);

void fncFdmtIteration_keith(int* d_piarrInp, const float val_dF, const int IDim0, const int IDim1
	, const int IDim2, const int IMaxDT, const float VAlFmin
	, const float VAlFmax, const int ITerNum, float* d_arr_val0
	, float* d_arr_val1, int* d_iarr_deltaTLocal
	, int* d_iarr_dT_MI, int* d_iarr_dT_ML, int* d_iarr_dT_RI
	, int* d_piarrOut, int& iOutPutDim0, int& iOutPutDim1);

__global__
void create_auxillary_1d_arrays_keith(const int IFjumps, const int IMaxDT, const float VAlTemp1
	, const float VAlc2, const float VAlf_min, const float VAlcorrection
	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal);



void fnc_init_fdmt_keith(int* d_piarrImg, const int IImgrows, const int IImgcols
	, const int IDeltaTplus1, int* d_piarrOut);




__global__
void kernel3D_shift_and_sum_keith(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, float* d_arr_val0, float* d_arr_val1
	, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut);

__host__ __device__
void calc3AuxillaryVars_keith(int& ideltaTLocal, int& i_dT, int& iF, float& val0
	, float& val1, int& idT_middle_index, int& idT_middle_larger, int& idT_rest_index);

__global__
void kernel3D_shift_and_sum_keith1(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, float* d_arr_val0, float* d_arr_val1
	, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut);


__host__ __device__ int array4d_idx(int nw, int nx, int ny, int nz, int w, int x, int y, int z);


__global__
void  fdmt_initialise_kernel(int* d_piarrImg,
	int* d_piarrOut, const int delta_t, const  int max_dt, const int IImgrows, const int IImgcols);

__global__
void  fdmt_initialise_kernel2(int* d_piarrImg, int* d_piarrOut, int delta_t
	, int max_dt, const int IImgrows, const int IImgcols, bool count);







