//#include "FdmtCpuF_omp.h"
#include "FdmtCpuT_omp.h"
#include "HybridC_v0.h"

#include "StreamParams.h"
//#include "utilites.h"
#include <fftw3.h>
#include "Constants.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <array>
#include <vector>
#include "npy.hpp"




#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif




using namespace std;

int fncHybridScan(float* parrSucessImagesBuff, int* piarrNumSuccessfulChunks, float *parrCoherent_d, int& quantOfSuccessfulChunks, CStreamParams* pStreamPars)
{
	const int NumChunks = ((pStreamPars->m_numEnd - pStreamPars->m_numBegin) + pStreamPars->m_lenChunk - 1) / pStreamPars->m_lenChunk;
	complex<float>* pRawSignalCur = new complex<float>[pStreamPars->m_lenChunk];
	quantOfSuccessfulChunks = 0;

	// remains not readed elements
	int iremains = pStreamPars->m_lenarr;
	float val_coherent_d;
	for (int i = 0; i < NumChunks; ++i)
	{
		int length = (iremains < pStreamPars->m_lenChunk) ? iremains : pStreamPars->m_lenChunk;		

		fread(pRawSignalCur, sizeof(complex<float>), length, pStreamPars->m_stream);
		float* poutImage = nullptr;
		if (nullptr != parrSucessImagesBuff)
		{
			poutImage = &parrSucessImagesBuff[pStreamPars->m_n_p * (pStreamPars->m_lenChunk / pStreamPars->m_n_p) * quantOfSuccessfulChunks];
		}
		if (fncSearchForHybridDedispersion(poutImage, pRawSignalCur, length, pStreamPars->m_n_p
			, pStreamPars->m_D_max, pStreamPars->m_f_min, pStreamPars->m_f_max, pStreamPars->m_SigmaBound, val_coherent_d))
			
		{
			piarrNumSuccessfulChunks[quantOfSuccessfulChunks] = i;
			parrCoherent_d[quantOfSuccessfulChunks] = val_coherent_d;
			++quantOfSuccessfulChunks;
		}
		++(pStreamPars->m_numCurChunk);
		iremains -= pStreamPars->m_lenChunk;
	}
	// ZAGLUSHKA!!!!
	//piarrNumSuccessfulChunks[0] = 0;
	//quantOfSuccessfulChunks = 1;

	// !
	delete[]pRawSignalCur;
	return 0;
}
//-------------------------------------------------------------
bool createOutImageForFixedNumberChunk(float* poutputImage, int* pargmaxRow, int* pargmaxCol, float* pvalSNR
	,float** pparrOutSubImage, int *piQuantRowsPartImage,CStreamParams* pStreamPars	, const int numChunk, const float VAlCoherent_d)
{
	int iremains = pStreamPars->m_lenarr - numChunk * pStreamPars->m_lenChunk;
	int lengthChunk = (iremains < pStreamPars->m_lenChunk) ? iremains : pStreamPars->m_lenChunk;
	int icols = lengthChunk / pStreamPars->m_n_p;
	
	complex<float>* pRawSignalCur = new complex<float>[lengthChunk];

	fseek(pStreamPars->m_stream, numChunk * pStreamPars->m_lenChunk* sizeof(complex<float>), SEEK_CUR);
	fread(pRawSignalCur, sizeof(complex<float>), lengthChunk, pStreamPars->m_stream);

	bool bres = false;
	
	float valSigmaBound = pStreamPars->m_SigmaBound;
	// 1. create FFT
	complex<float>* pffted_rowsignal = (complex<float>*)malloc(sizeof(complex<float>) * lengthChunk);
	fftwf_complex* fftw_out = reinterpret_cast<fftwf_complex*>(pffted_rowsignal);
	fftwf_complex* fftw_in = reinterpret_cast<fftwf_complex*>(pRawSignalCur);
	// Create the FFT plan
	fftwf_plan plan = fftwf_plan_dft_1d(lengthChunk, fftw_in, fftw_out, FFTW_FORWARD, FFTW_ESTIMATE);

	// Execute the FFT
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);

	// !1

		// 2. create fdmt ones
	float* parr_fdmt_ones = (float*)malloc(lengthChunk * sizeof(float));
	for (int i = 0; i < lengthChunk; ++i)
	{
		parr_fdmt_ones[i] = 1.;
	}
	int IMaxDT = pStreamPars->m_n_p;   
	int N_p = pStreamPars->m_n_p;
	const float VAlFmin = pStreamPars->m_f_min;
	const float VAlFmax = pStreamPars->m_f_max;
	const float VAlD_max = pStreamPars->m_D_max;
	float* parrImNormalize = (float*)malloc(N_p * (lengthChunk / N_p) * sizeof(float));


	fncFdmt_cpuT_v0(parr_fdmt_ones, N_p, lengthChunk / N_p
		, VAlFmin, VAlFmax, IMaxDT, parrImNormalize);


	free(parr_fdmt_ones);
	// !2

	// 3.		
	float valConversionConst = DISPERSION_CONSTANT * (1. / (VAlFmin * VAlFmin) - 1. / (VAlFmax * VAlFmax)) * (VAlFmax - VAlFmin);
	float valN_d = VAlD_max * valConversionConst;
	

	// !3

	//4.
	complex<float>* pcarrCD_Out = (complex<float>*)malloc(sizeof(complex<float>) * lengthChunk);
	complex<float>* pcarrTemp = (complex<float>*)malloc(sizeof(complex<float>) * (lengthChunk / N_p) * N_p);
	float* parr_fdmt_inp = (float*)malloc(sizeof(float) * (lengthChunk / N_p) * N_p);
		

		
		createOutputFDMT(poutputImage, pffted_rowsignal, pcarrCD_Out, pcarrTemp
			, lengthChunk, N_p, parr_fdmt_inp, IMaxDT
			, DISPERSION_CONSTANT * VAlCoherent_d, VAlD_max, VAlFmin, VAlFmax);
		int len = (lengthChunk / N_p) * N_p;

		float maxSig0 = -1., maxSig1 = -1.;
		int argmax = -1;
		*pvalSNR = (*max_element(poutputImage, poutputImage + len));
		argmax = max_element(poutputImage, poutputImage + len) - poutputImage;
		*pargmaxRow = argmax / (lengthChunk / N_p);
		*pargmaxCol = argmax % (lengthChunk / N_p);

		cutQuadraticSubImage(pparrOutSubImage, piQuantRowsPartImage, poutputImage, N_p, (lengthChunk / N_p), *pargmaxRow, *pargmaxCol);

	
	free(pcarrTemp);
	free(pcarrCD_Out);
	free(parr_fdmt_inp);
	
	free(parrImNormalize);
	return bres;

}
//-----------------------------------------------------------------------------------------
void cutQuadraticSubImage(float** pparrOutImage, int *piQuantRowsOutImage,float* InpImage, const int QInpImageRows, const int QInpImageCols
	, const int NUmCentralElemRow, const int NUmCentralElemCol)
{
	*piQuantRowsOutImage = (QInpImageRows < QInpImageCols) ? QInpImageRows : QInpImageCols;
	(*pparrOutImage) = (float*)realloc((*pparrOutImage), (*piQuantRowsOutImage) * (*piQuantRowsOutImage) * sizeof(float));
	float* p = (*pparrOutImage);
	if (QInpImageRows < QInpImageCols)
	{
		int numPart = NUmCentralElemCol / QInpImageRows;
		int numColStart = numPart * QInpImageRows;
		for (int i = 0; i < QInpImageRows; ++i)
		{
			memcpy(&p[i * QInpImageRows], &InpImage[i * QInpImageCols + numColStart], QInpImageRows * sizeof(float));
		}
		return;
    }
	int numPart = NUmCentralElemRow / QInpImageCols;
	int numStart = numPart * QInpImageCols;
	memcpy(p, &InpImage[numStart], QInpImageCols * QInpImageCols * sizeof(float));
}

//--------------------------------------------------------------
//INPUT:
// pffted_rowsignal - complex array, ffted 1-dimentional row signal, done from current chunk,  length = LEnChunk
// pcarrCD_Out - memory allocated comlex buffer to save output of coherent dedispersion function, nmed as fncCoherentDedispersion,
//				1- dimentional complex array, length = 	LEnChunk
// pcarrTemp - memory allocated comlex buffer to save output of STFT function, named as fncSTFT. 2-dimentional complex array
//            with dimensions = N_p x (LEnChunk / N_p)
// LEnChunk - length of input ffted signal pffted_rowsignal
// N_p - 
// parr_fdmt_inp - memory allocated float buffer to save input for FDMT function, dimentions = N_p x (LEnChunk / N_p)
// IMaxDT - the maximal delay (in time bins) of the maximal dispersion. Appears in the paper as N_{\Delta}
//            A typical input is maxDT = N_f
// VAlLong_coherent_d - is DispersionConstant* d, where d - is the dispersion measure.units: pc * cm ^ -3
// VAlD_max - maximal dispersion to scan, in units of pc cm^-3
// VAlFmin - the minimum freq, given in Mhz
// VAlFmax - the maximum freq,
//
// OUTPUT:
// parr_fdmt_out - float 2-dimensional array,with dimensions =  IMaxDT x (LEnChunk / N_p)
int createOutputFDMT(float* parr_fdmt_out, complex<float>* pffted_rowsignal,  complex<float>* pcarrCD_Out, complex<float>* pcarrTemp
	, const unsigned int LEnChunk, const unsigned int N_p, float * parr_fdmt_inp, const unsigned int IMaxDT
	, const long double VAlLong_coherent_d	, const float VAlD_max, const float VAlFmin, const float VAlFmax)
{

	fncCoherentDedispersion(pcarrCD_Out, pffted_rowsignal
		, LEnChunk, VAlLong_coherent_d, VAlFmin, VAlFmax);

	fncSTFT(pcarrTemp, pcarrCD_Out, LEnChunk, N_p);

	float sum = 0.;
	int len = (LEnChunk / N_p) * N_p;
	for (int j = 0; j < len; ++j)
	{
		float temp = pcarrTemp[j].real() * pcarrTemp[j].real() + pcarrTemp[j].imag() * pcarrTemp[j].imag();

		sum += temp;
		parr_fdmt_inp[j] = temp;
	}

	float val_mean = sum / ((float)len);
	float valStdDev = fnsStdDev(parr_fdmt_inp, val_mean, len);

	for (int j = 0; j < len; ++j)
	{
		parr_fdmt_inp[j] = (parr_fdmt_inp[j] - val_mean) / (0.25 * valStdDev);
	}	

	fncFdmt_cpuT_v0(parr_fdmt_inp, N_p, LEnChunk / N_p
		, VAlFmin, VAlFmax, IMaxDT, parr_fdmt_out);

	return 0;
}
//---------------------------------------------------------------------------------------------------------
void fncMaxSignalDetection(float* parr_fdmt_out, float* parrImNormalize, const unsigned int qRows, const unsigned int qCols
	, float* pmaxElement, int* argmax)

{
	const unsigned int len = qRows * qCols;
	
	
	float* p = parr_fdmt_out;
	float* pn = parrImNormalize;
	for (int i = 0; i < len; ++i)
	{
		*p = (*p) / sqrt(((*pn) * 16. + 0.000001));
		++p;
		++pn;
	}

	//float* pmaxElement1 = max_element(parr_fdmt_out, parr_fdmt_out + len);
	(*pmaxElement) = (*max_element(parr_fdmt_out, parr_fdmt_out + len));
	*argmax = max_element(parr_fdmt_out, parr_fdmt_out + len) - parr_fdmt_out;
}

//-----------------------------------------------------------
void fncSTFT(complex<float>* pcarrOut, complex<float>* pRawSignalCur,  const unsigned int LEnChunk, int block_size)
{
	int qRows = LEnChunk / block_size;
	// allocate memory for temporary matrix
	complex<float>* pcarrS0 = (complex<float>*)malloc(sizeof(complex<float>) * qRows * block_size);

	fftwf_complex* fftw_out = reinterpret_cast<fftwf_complex*>(pcarrS0);
	fftwf_complex* fftw_in = reinterpret_cast<fftwf_complex*>(pRawSignalCur);
	for (int i = 0; i < qRows; ++i)
	{
		fftwf_plan plan = fftwf_plan_dft_1d(block_size, &fftw_in[i * block_size], &fftw_out[i * block_size], FFTW_FORWARD, FFTW_ESTIMATE);
		fftwf_execute(plan);
		fftwf_destroy_plan(plan);
		
	}
	

	fncMtrxTranspose(pcarrOut, pcarrS0, qRows, block_size);
	
	free(pcarrS0);	
}

//---------------------------------------------------------------------------------
void fncCoherentDedispersion(complex<float>* pcarrCD_Out, complex<float>* pcarrffted_rowsignal
	, const unsigned int LEnChunk, const long double VAl_practicalD, const float VAlFmin, const float VAlFmax)
{
	long double * pcarrf = (long double*)malloc(LEnChunk * sizeof(long double));
	long double step = ((long double)VAlFmax - (long double)VAlFmin) / ((long double)LEnChunk);
	for (int i = 0; i < LEnChunk; ++i)
	{
		pcarrf[i] = ((long double)i) * step;
		
	}

	complex<long double>* pcarrH = (complex<long double>*)malloc(LEnChunk * sizeof(complex<long double>));
	
	complex <float> cmp_temp (0.,   -2. * M_PI * VAl_practicalD) ;
	//float val_fcur = 0.;
	//H = np.e * *(-(2 * np.pi * complex(0, 1) * practicalD / (f_min + f) + 2 * np.pi * complex(0, 1) * practicalD * f / (f_max * *2)))
	for (int i = 0; i < LEnChunk; ++i)
	{
		//float valt0 = -VAl_practicalD / (VAlFmin + pcarrf[i]) * 2.* M_PI;
		//float valt1 = -VAl_practicalD / (VAlFmax * VAlFmax) * pcarrf[i] * 2. * M_PI;
		long double t0 = fmodl((VAl_practicalD / ((long double)VAlFmin + (long double)pcarrf[i]) + VAl_practicalD / ((long double)VAlFmax * (long double)VAlFmax) * (long double)pcarrf[i]) * 2. * M_PI
			, 2. * M_PI); 

		//long double fractionalPart = t0 - floor(t0);

		//complex <double> cmp_temp1(0., -fractionalPart * 2. * M_PI);
		//	cmp_temp / complex < float>(VAl_practicalD, 0) + cmp_temp * complex < float>(pcarrf[i] / (VAlFmax * VAlFmax), 0.);
		complex<long double> t1(0., -t0);
		long double t3 = (VAl_practicalD / ((long double)VAlFmin + (long double)pcarrf[i])
			+ VAl_practicalD / ((long double)VAlFmax * (long double)VAlFmax) * (long double)pcarrf[i]) * 2. * M_PI;
		complex<long double> t2(0., -t3);
		pcarrH[i] = exp(t2);
		//val_fcur += step;
	}	

	
	complex<float>* pcarrTemp = (complex<float>*)malloc(LEnChunk * sizeof(complex<float>));
	for (int i = 0; i < LEnChunk; ++i)
	{
		pcarrTemp[i] = (complex<float>)pcarrH[i] * pcarrffted_rowsignal[i];
	}
	free(pcarrH);
	free(pcarrf);


	
	fftwf_complex* fftw_in = reinterpret_cast<fftwf_complex*>(pcarrTemp);
	fftwf_complex* fftw_out = reinterpret_cast<fftwf_complex*>(pcarrCD_Out);
	// Create the FFT plan

	fftwf_plan plan = fftwf_plan_dft_1d(LEnChunk, fftw_in, fftw_out, FFTW_BACKWARD, FFTW_ESTIMATE);
	
	// Execute the FFT
	fftwf_execute(plan);
	complex<float> cmp_temp1((float)LEnChunk, 0.);
	for (int i = 0; i < LEnChunk; ++i)
	{
		pcarrCD_Out[i] = pcarrCD_Out[i] / cmp_temp1;
	}
	
	fftwf_destroy_plan(plan);
	free(pcarrTemp);

}

//-----------------------------------------------------------------------------------------------------------------

bool fncSearchForHybridDedispersion(float* poutImage, complex<float>* pRawSignalCur
	, const unsigned int LEnChunk, const unsigned int N_p
	, const float VAlD_max, const float VAlFmin, const float VAlFmax, float& valSigmaBound_, float& coherent_d)
{
	bool bres = false;
	coherent_d = -1.;
	float valSigmaBound = valSigmaBound_;
	// 1. create FFT
	complex<float>* pffted_rowsignal = (complex<float>*)malloc(sizeof(complex<float>) * LEnChunk);
	fftwf_complex* fftw_out = reinterpret_cast<fftwf_complex*>(pffted_rowsignal);
	fftwf_complex* fftw_in = reinterpret_cast<fftwf_complex*>(pRawSignalCur);
	// Create the FFT plan
	fftwf_plan plan = fftwf_plan_dft_1d(LEnChunk, fftw_in, fftw_out, FFTW_FORWARD, FFTW_ESTIMATE);

	// Execute the FFT
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);

	// !1

		// 2. create fdmt ones
	float* parr_fdmt_ones = (float*)malloc(LEnChunk * sizeof(float));
	for (int i = 0; i < LEnChunk; ++i)
	{
		parr_fdmt_ones[i] = 1.;
	}
	int IMaxDT = N_p;
	float* parrImNormalize = (float*)malloc(N_p * (LEnChunk / N_p) * sizeof(float));

	//fncFdmt_cpuF_v0(parr_fdmt_ones, N_p, LEnChunk / N_p
	//	, VAlFmin, VAlFmax, IMaxDT, parrImNormalize);


	fncFdmt_cpuT_v0(parr_fdmt_ones, N_p, LEnChunk / N_p
		, VAlFmin, VAlFmax, IMaxDT, parrImNormalize);


	free(parr_fdmt_ones);
	// !2

	// 3.		
	float valConversionConst = DISPERSION_CONSTANT * (1. / (VAlFmin * VAlFmin) - 1. / (VAlFmax * VAlFmax)) * (VAlFmax - VAlFmin);
	float valN_d = VAlD_max * valConversionConst;
	int n_coherent = int(ceil(valN_d / (N_p * N_p)));
	cout << "n_coherent = " << n_coherent << endl;

	// !3

	//4.
	complex<float>* pcarrCD_Out = (complex<float>*)malloc(sizeof(complex<float>) * LEnChunk);
	complex<float>* pcarrTemp = (complex<float>*)malloc(sizeof(complex<float>) * (LEnChunk / N_p) * N_p);
	float* parr_fdmt_inp = (float*)malloc(sizeof(float) * (LEnChunk / N_p) * N_p);
	float* parr_fdmt_out = (float*)malloc(sizeof(float) * (LEnChunk / N_p) * N_p);
	
	for (int iouter_d = 31; iouter_d < 32; ++iouter_d)
		//for (int iouter_d = 0; iouter_d < n_coherent; ++iouter_d)
	{
		cout << "coherent iteration " << iouter_d << endl;

		long double valcur_coherent_d = ((long double)iouter_d) * ((long double)VAlD_max / ((long double)n_coherent));
		cout << "cur_coherent_d = " << valcur_coherent_d << endl;

		createOutputFDMT(parr_fdmt_out, pffted_rowsignal, pcarrCD_Out, pcarrTemp
			, LEnChunk, N_p, parr_fdmt_inp, IMaxDT
			, DISPERSION_CONSTANT * valcur_coherent_d, VAlD_max, VAlFmin, VAlFmax);
		int len = (LEnChunk / N_p) * N_p;

		float maxSig= -1.;
		int iargmax =  -1;
		fncMaxSignalDetection(parr_fdmt_out, parrImNormalize, N_p, (LEnChunk / N_p)
			, &maxSig,  &iargmax);

		if (maxSig > valSigmaBound)
		{
			valSigmaBound = maxSig;
			coherent_d = valcur_coherent_d;
			cout << "!!!!!!! achieved score with " << valSigmaBound << "!!!!!!!" << endl;
			bres = true;
			if (nullptr != poutImage)
			{
				memcpy(poutImage, parr_fdmt_out, len * sizeof(float));
			}
			std::cout << "SNR = " << maxSig << endl;
			std::cout << "NUM ARGMAX = " << iargmax << endl;
			std::cout << "ROW ARGMAX = " << (int)(iargmax/(LEnChunk / N_p)) << endl;
			std::cout << "COLUMN ARGMAX = " << (int)(iargmax %(LEnChunk / N_p)) << endl;

		}
	}
	free(pcarrTemp);
	free(pcarrCD_Out);
	free(parr_fdmt_inp);
	free(parr_fdmt_out);
	free(parrImNormalize);
	return bres;
}

//----------------------------------------------------------------
template <typename T>
void fncMtrxTranspose(T*pArrout, T* pArrinp, const int QRowsInp, const int QColsInp)
{
	T* pint = pArrinp;
	T* pout = pArrout;
	for (int i = 0; i < QRowsInp; ++i)
	{
		pout = pArrout + i;
		for (int j = 0; j < QColsInp; ++j)
		{
			*pout = *pint;
			++pint;
			pout += QRowsInp;
		}

	}
}
//-------------------------------------------------
void fncElementWiseModSq(float* parrOut, complex<float>* pcarrInp, unsigned int len)
{
	for (int i = 0; i < len; ++i)
	{
		parrOut[i] = pcarrInp[i].real() * pcarrInp[i].real() + pcarrInp[i].imag() * pcarrInp[i].imag();
	}
}
//-------------------------------------------------------------------
template <typename T>
float fnsStdDev(T* parr_fdmt_inp, const float mean, unsigned int len)
{
	float sum = 0.;
	for (int i = 0; i < len; ++i)
	{
		sum += ((float)parr_fdmt_inp[i] - mean) * ((float)parr_fdmt_inp[i] - mean);
	}
	return sqrt(sum/ ((float)len));
}
//------------------------------------------------------------------
template <typename T>
void fncFdmt_cpuT_v1(T* piarrImg, const int iImgrows
	, const int iImgcols, const float f_min
	, const  float f_max, const int imaxDT, T* piarrOut)
{
	
}
//-------------------------------------------------------------
template <typename T>
void fncDisp(T* parr_fdmt_inp, unsigned int len, T& val_mean, T& val_V)
{
	val_mean = 0;
	T* p = parr_fdmt_inp;
	for (int i = 0; i < len; ++i)
	{
		val_mean += *p;
		++p;
	}

	val_mean = val_mean / len;

	val_V = 0;
	p = parr_fdmt_inp;
	for (int i = 0; i < len; ++i)
	{
		T temp = *p - val_mean;
		val_V += temp * temp;
		++p;
	}
	val_V /= ((T)len);

}