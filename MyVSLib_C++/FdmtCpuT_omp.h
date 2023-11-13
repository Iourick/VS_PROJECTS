#pragma once
//-------------------------------------------------------------------------
template <typename T>
void fncFdmt_cpuT_v0(T* piarrImg, const int iImgrows
	, const int iImgcols, const float f_min
	, const  float f_max, const int imaxDT, T* piarrOut);

template <typename T>
void fnc_init_cpuT(T* piarrImg, const int IImgrows, const int IImgcols
	, const int IDeltaT, T* piarrOut);

template <typename T>
void fncFdmtIteration_cpuT(T* piarrInp, const float val_dF, const int IDim0, const int IDim1
	, const int IDim2, const int IMaxDT, const float VAlFmin
	, const float VAlFmax, const int ITerNum, float* arr_val0
	, float* arr_val1, int* iarr_deltaTLocal
	, int* iarr_dT_MI, int* iarr_dT_ML, int* iarr_dT_RI
	, T* piarrOut, int& iOutPutDim0, int& iOutPutDim1);


void create_2d_arrays_cpu(const int IDim0, const int IDim1
	, float* arr_val0, float* arr_val1, int* iarr_deltaTLocal
	, int* iarr_dT_middle_index, int* iarr_dT_middle_larger
	, int* iarr_dT_rest_index);

template <typename T>
void shift_and_sum_cpuT_v1(T* piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* iarr_deltaTLocal, int* iarr_dT_MI
	, int* iarr_dT_ML, int* iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, T* piarrOut);

template <typename T>
void shift_and_sum_cpuT(T* piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* iarr_deltaTLocal, int* iarr_dT_MI
	, int* iarr_dT_ML, int* iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, T* piarrOut);




