void fncFdmt_cu_v0(int* piarrImg, const int iImgrows
	, const int iImgcols, const float f_min
	, const  float f_max, const int imaxDT, int* piarrOut);

void fnc_init(int* d_piarrImg, const float f_min, const  float f_max
	, const int imaxDT, const int IImgrows, const int IImgcols
	, const int IDimOut1, int* d_piarrOut);

void fncFdmtIteration(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, const int IMaxDT, const float VAlFmin
	, const float VAlFmax, const int ITerNum, float* d_arr_val0
	, float* d_arr_val1, int* d_arr_deltaTLocal
	, int* d_arr_dT_MI, int* d_arr_dT_ML, int* d_arr_dT_RI
	,int* d_piarrOut);

__global__
void create_auxillary_1d_arrays(const int IFjumps, const int IMaxDT, const float VAlTemp1
	, const float VAlc2, const float VAlf_min, const float VAlcorrection
	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal);

__global__
void create_auxillary_2d_arrays(const int IDim0, const int IDim1
	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal
	, int* d_iarr_dT_middle_index, int* d_iarr_dT_middle_larger
	, int* d_arr_dT_rest_index);

__global__
void kernel_shift(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI
	, int* d_piarrOut);