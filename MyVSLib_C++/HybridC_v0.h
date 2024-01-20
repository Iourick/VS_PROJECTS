#pragma once
#include "FdmtCpuT_omp.h"

#include <fftw3.h>
#include <complex>

using namespace std;

class CStreamParams;
int fncHybridScan(float* parrSucessImagesBuff, int* piNumSuccessfulChunks, float* parrCoherent_d, int& quantOfSuccessfulChunks, CStreamParams* pStreamPars);

bool fncSearchForHybridDedispersion(float* poutImage, fftwf_complex* pRawSignalCur
	, const unsigned int LEnChunk, const unsigned int N_p
	, const float VAlD_max, const float VAlFmin, const float VAlFmax, float& valSigmaBound_, float& coherent_d);

bool createOutImageForFixedNumberChunk(float* poutputImage, int* pargmaxRow, int* pargmaxCol, float* pvalSNR
	, float** pparrOutSubImage, int* piQuantRowsPartImage, CStreamParams* pStreamPars, const int numChunk
	, const float VAlCoherent_d);

template <typename T>
void fncMtrxTranspose(T* pArrout, T* pArrinp, const int QRowsInp, const int QColsInp);

void fncSTFT(fftwf_complex* pcarrOut, fftwf_complex* pRawSignalCur, const unsigned int LEnChunk, int block_size);

void fncElementWiseModSq(float* parrOut, fftwf_complex* pcarrInp, unsigned int len);

void fncCoherentDedispersion(fftwf_complex* pcarrCD_Out, fftwf_complex* pcarrffted_rowsignal
	, const unsigned int LEnChunk, const long double VAl_practicalD, const float VAlFmin, const float VAlFmax);

template <typename T>
float fnsStdDev(T* parr_fdmt_inp, const float mean, unsigned int len);

template <typename T>
void fncFdmt_cpuT_v1(T* piarrImg, const int iImgrows
	, const int iImgcols, const float f_min
	, const  float f_max, const int imaxDT, T* piarrOut);

template <typename T>
void fncDisp(T* parr_fdmt_inp, unsigned int len, T& val_mean, T& val_V);

int createOutputFDMT(float* parr_fdmt_out, fftwf_complex* pffted_rowsignal, fftwf_complex* pcarrCD_Out, fftwf_complex* pcarrTemp
	, const unsigned int LEnChunk, const unsigned int N_p, float* parr_fdmt_inp, const unsigned int IMaxDT
	, const long double VAlLong_coherent_d, const float VAlD_max, const float VAlFmin, const float VAlFmax);

void fncMaxSignalDetection(float* parr_fdmt_out, float* parrImNormalize, const unsigned int qRows, const unsigned int qCols
	, float* pmaxElement, int* argmax);

void cutQuadraticSubImage(float** pparrOutImage, int* piQuantRowsOutImage, float* InpImage, const int QInpImageRows, const int QInpImageCols
	, const int NUmCentralElemRow, const int NUmCentralElemCol);

void fncMtrxTranspose_(fftwf_complex* pArrout, fftwf_complex* pArrinp, const int QRowsInp, const int QColsInp);