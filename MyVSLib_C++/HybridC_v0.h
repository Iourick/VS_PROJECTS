#pragma once
#include "FdmtCpuT_omp.h"
#include <complex>

using namespace std;

class CStreamParams;
int fncHybridScan(int* piNumSuccessfulChunks, float* parrCoherent_d, int& quantOfSuccessfulChunks, CStreamParams* pStreamPars);

int fncHybridDedispersion(float* poutImage, std::complex<float>* pRawSignalCur, const unsigned int LEnChunk, const unsigned int N_p
	, const float VAlD_max, const float VAlf_min, const float VAlf_max,  float &VAlSigmaBound, float& coherent_d);

void createOutImageForFixedNumberChunk(float* outputImage, CStreamParams* pStreamPars
	, const int numChunk);

template <typename T>
void fncMtrxTranspose(T* pArrout, T* pArrinp, const int QRowsInp, const int QColsInp);

void fncSTFT(complex<float>* pcarrOut, complex<float>* pRawSignalCur, const unsigned int LEnChunk, int block_size);

void fncElementWiseModSq(float* parrOut, complex<float>* pcarrInp, unsigned int len);



void fncCoherentDedispersion(complex<float>* pcarrCD_Out, complex<float>* pcarrffted_rowsignal
	, const unsigned int LEnChunk, const float VAl_practicalD, const float VAlFmin, const float VAlFmax);

template <typename T>
float fnsStdDev(T* parr_fdmt_inp, const float mean, unsigned int len);

template <typename T>
void fncFdmt_cpuT_v1(T* piarrImg, const int iImgrows
	, const int iImgcols, const float f_min
	, const  float f_max, const int imaxDT, T* piarrOut);

template <typename T>
void fncDisp(T* parr_fdmt_inp, unsigned int len, T& val_mean, T& val_V);