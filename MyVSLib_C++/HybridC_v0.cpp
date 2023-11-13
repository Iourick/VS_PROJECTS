#include "FdmtCpuF_omp.h"
#include "HybridC_v0.h"

#include "StreamParams.h"
//#include "utilites.h"
#include <fftw3.h>
#include "Constants.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>


#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif


/// <summary>
/// 
/// </summary>

using namespace std;

int fncHybridScan(int* piNumSuccessfulChunks, float *parrCoherent_d, int& quantOfSuccessfulChunks, CStreamParams* pStreamPars)
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

		
		if (fncHybridDedispersion(nullptr,   pRawSignalCur, length, pStreamPars->m_n_p
			, pStreamPars->m_D_max, pStreamPars->m_f_min, pStreamPars->m_f_max, pStreamPars->m_SigmaBound, val_coherent_d)
			== 0)
		{
			piNumSuccessfulChunks[quantOfSuccessfulChunks] = i;
			parrCoherent_d[quantOfSuccessfulChunks] = val_coherent_d;
			++quantOfSuccessfulChunks;
		}
		++(pStreamPars->m_numCurChunk);
		iremains -= pStreamPars->m_lenChunk;
	}
	// ZAGLUSHKA!!!!
	piNumSuccessfulChunks[0] = 0;
	quantOfSuccessfulChunks = 1;

	// !
	delete[]pRawSignalCur;
	return 0;
}
//-----------------------------------------------------------------------------------------

int fncHybridDedispersion(float *poutImage, complex<float>* pRawSignalCur, const unsigned int LEnChunk, const unsigned int N_p
	, const float VAlD_max, const float VAlFmin, const float VAlFmax, float &valSigmaBound, float &coherent_d )
{
	coherent_d = -1.;
	/*ConversionConst = DispersionConstant * (1. / f_min * *2 - 1. / f_max * *2) * (f_max - f_min)
		N_d = D_max * ConversionConst

		n_coherent = int(np.ceil(N_d / (N_p * *2)))
		print("number of coherent iterations:", n_coherent)
		ffted_signal = np.fft.fft(raw_signal)*/
	// 1. create FFT
		complex<float>* pffted_rowsignal = (complex<float>*)malloc(sizeof(complex<float>) * LEnChunk);
		fftwf_complex* fftw_out = reinterpret_cast<fftwf_complex*>(pffted_rowsignal);
		fftwf_complex* fftw_in = reinterpret_cast<fftwf_complex*>(pRawSignalCur);
		// Create the FFT plan
		fftwf_plan plan = fftwf_plan_dft_1d(LEnChunk, fftw_in, fftw_out, FFTW_FORWARD, FFTW_ESTIMATE);

		// Execute the FFT
		fftwf_execute(plan);
	// !1

		// 2. create fdmt ones
		float* parr_fdmt_ones = (float*)malloc(LEnChunk  * sizeof(float));
		for (int i = 0; i < LEnChunk; ++i)
		{
			parr_fdmt_ones[i] = 1.;
		}
		int IMaxDT = N_p;
		float* piarrImNormalize = (float*)malloc(N_p * (LEnChunk/ N_p) * sizeof(float));

		

		fncFdmt_cpuF_v0(parr_fdmt_ones, N_p, LEnChunk/N_p
			, VAlFmin, VAlFmax, IMaxDT, piarrImNormalize);
		free(parr_fdmt_ones);
		// !2

		// 3.		
		float valConversionConst = DISPERSION_CONSTANT * (1. / (VAlFmin * VAlFmin) - 1. / (VAlFmax * VAlFmax)) * (VAlFmax - VAlFmin);
		float valN_d = VAlD_max * valConversionConst;
		int n_coherent = int(ceil(valN_d / (N_p * N_p)));
		// !3

		//4.
		complex<float>* pcarrCD_Out = (complex<float>*)malloc(sizeof(complex<float>) * LEnChunk);
		complex<float>* pcarrTemp = (complex<float>*)malloc(sizeof(complex<float>) * (LEnChunk/ N_p) * N_p);
		float* parr_fdmt_inp = (float*)malloc(sizeof(float) * (LEnChunk / N_p) * N_p);
		float* parr_fdmt_out = (float*)malloc(sizeof(float) * (LEnChunk / N_p) * N_p);
		for (int iouter_d = 0; iouter_d < n_coherent; ++iouter_d)
		{
			float valcur_coherent_d = iouter_d * (VAlD_max / ((float)n_coherent));
			fncCoherentDedispersion(pcarrCD_Out, pffted_rowsignal
				, LEnChunk, DISPERSION_CONSTANT * valcur_coherent_d, VAlFmin, VAlFmax);
			fncSTFT(pcarrTemp, pcarrCD_Out, LEnChunk, N_p);
			
			float sum = 0.;
			int len = (LEnChunk / N_p) * N_p;
			for (int j = 0; j < len ; ++j)
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

			
			fncFdmt_cpuF_v0(parr_fdmt_inp, N_p, LEnChunk/ N_p
				, VAlFmin, VAlFmax, IMaxDT, parr_fdmt_out);
			float val_V = 0., val_mean1 = 0.;
			fncDisp(parr_fdmt_inp, len, val_mean1,val_V);
			float* p = parr_fdmt_inp;
			float* pn = piarrImNormalize;
			for (int i = 0; i < len; ++i)
			{
				*p = (*p) / ((*pn) * val_V + 0.000001);
				++p;
				++pn;
			}

			float* pmaxElement = std::max_element(parr_fdmt_inp, parr_fdmt_inp + len);
			if ((*pmaxElement) > valSigmaBound)
			{
				valSigmaBound = (*pmaxElement);
				coherent_d = valcur_coherent_d;
				cout <<"achieved score with " << valSigmaBound << endl;
				if (nullptr != poutImage)
				{
					memcpy(poutImage, parr_fdmt_inp, len * sizeof(float));
				}

			}			

		}
		free(pcarrTemp);
		free(pcarrCD_Out);
		free(parr_fdmt_inp);
		free(parr_fdmt_out);
		free(piarrImNormalize);
	return 0;
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
		fftwf_plan plan = fftwf_plan_dft_1d(block_size, fftw_in, fftw_out, FFTW_FORWARD, FFTW_ESTIMATE);
		fftwf_execute(plan);
		//fftw_destroy_plan(plan);
		fftw_in += block_size;
		fftw_out += block_size;
	}

	fncMtrxTranspose<complex<float>>(pcarrOut, pcarrS0, qRows, block_size);

	free(pcarrS0);

	
}
//def STFT :
//	"""
//	Raw signal will be devided to blocks, each block will be fourier transformed
//	Input :
//raw_signal - raw antenna voltage time series
//block_size - number of bins in each block
//Output :
//frequency vs.time matrix
//Note : absolute value squared is not performed!
//"""
//S = np.transpose(raw_signal[:int(len(raw_signal)//block_size) * block_size].reshape([int(len(raw_signal)//block_size),block_size]))
//	return np.fft.fft(S, axis = 0)

//---------------------------------------------------------------------------------
void fncCoherentDedispersion(complex<float>* pcarrCD_Out, complex<float>* pcarrffted_rowsignal
	, const unsigned int LEnChunk, const float VAl_practicalD, const float VAlFmin, const float VAlFmax)
{

	complex<float>* pcarrH = (complex<float>*)malloc(LEnChunk * sizeof(complex<float>));
	float step = (VAlFmax - VAlFmin) / ((float)LEnChunk);
	complex <float> cmp_temp = complex < float>(2. * M_PI * VAl_practicalD, 0) * complex<float>(0, 1);
	float val_fcur = 0.;
	for (int i = 0; i < LEnChunk; ++i)
	{
		pcarrH[i] = exp(cmp_temp / complex < float>(VAlFmin + val_fcur, 0) + cmp_temp * complex < float>(val_fcur / (VAlFmax * VAlFmax), 0.) );
		val_fcur += step;
	}

	complex<float>* pcarrTemp = (complex<float>*)malloc(LEnChunk * sizeof(complex<float>));
	for (int i = 0; i < LEnChunk; ++i)
	{
		pcarrTemp[i] = pcarrH[i] * pcarrffted_rowsignal[i];
	}
	free(pcarrH);


	fftwf_complex* fftw_in = reinterpret_cast<fftwf_complex*>(pcarrTemp);
	fftwf_complex* fftw_out = reinterpret_cast<fftwf_complex*>(pcarrCD_Out);
	// Create the FFT plan

	fftwf_plan plan = fftwf_plan_dft_1d(LEnChunk, fftw_in, fftw_out, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Execute the FFT
	fftwf_execute(plan);
	free(pcarrTemp);

	//def CoherentDedispersion(raw_signal, d, f_min, f_max, alreadyFFTed = False) :
	//	"""
	//	Will perform coherent dedispersion.
	//	raw signal - is assumed to be a one domensional signal
	//	d - is the dispersion measure.units : pc * cm ^ -3
	//	f_min - the minimum freq, given in Mhz
	//	f_max - the maximum freq, given in Mhz
	//	alreadyFFTed - to reduce complexity, insert fft(raw_signal) instead of raw_signal, and indicate by this flag
	//
	//	For future improvements :
	//1) Signal partition to chunks of length N_d is not applied, and maybe it should be.
	//2) No use of packing is done, though it is obvious it should be done(either in the coherent stage(and take special care of the abs()** 2 operation done by other functions) or in the incoherent stage)
	//
	//"""
	//N_total = len(raw_signal)
	//practicalD = DispersionConstant * d
	//f = np.arange(0, f_max - f_min, float(f_max - f_min) / N_total)
	//
	//# The added linear term makes the arrival times of the highest frequencies be 0
	//H = np.e * *(-(2 * np.pi * complex(0, 1) * practicalD / (f_min + f) + 2 * np.pi * complex(0, 1) * practicalD * f / (f_max * *2)))
	//if not alreadyFFTed:
	//CD = np.fft.ifft(np.fft.fft(raw_signal) * H)
	//else :
	//CD = np.fft.ifft(raw_signal * H)
	//return CD
}
//-------------------------------------------------------------
void createOutImageForFixedNumberChunk(float* outputImage, CStreamParams* pStreamPars
	, const int numChunk)
{
	int iremains = pStreamPars->m_lenarr - numChunk * pStreamPars->m_lenChunk;
	int length = (iremains < pStreamPars->m_lenChunk) ? iremains : pStreamPars->m_lenChunk;
	int icols = length / pStreamPars->m_n_p;
	memset(outputImage, 0, (pStreamPars->m_n_p) * icols * sizeof(float));
	for (int j = 0; j < icols; ++j)
	{
		outputImage[j] = 128.;
		outputImage[50 * icols + j] = 220.;
	}


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
float fnsStdDev(float* parr_fdmt_inp, float mean, unsigned int len)
{
	float sum = 0.;
	for (int i = 0; i < len; ++i)
	{
		sum += (parr_fdmt_inp[i] - mean) * (parr_fdmt_inp[i] - mean);
	}
	return sqrt(sum);
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

}