#include "FdmtCpuF_omp.h"
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
	
	// 1. create FFT
		complex<float>* pffted_rowsignal = (complex<float>*)malloc(sizeof(complex<float>) * LEnChunk);
		fftwf_complex* fftw_out = reinterpret_cast<fftwf_complex*>(pffted_rowsignal);
		fftwf_complex* fftw_in = reinterpret_cast<fftwf_complex*>(pRawSignalCur);
		// Create the FFT plan
		fftwf_plan plan = fftwf_plan_dft_1d(LEnChunk, fftw_in, fftw_out, FFTW_FORWARD, FFTW_ESTIMATE);

		// Execute the FFT
		fftwf_execute(plan);
		fftwf_destroy_plan(plan);


		// !!!
		std::vector<complex<float>> v2(pffted_rowsignal, pffted_rowsignal + LEnChunk);

		std::array<long unsigned, 1> leshape2{ LEnChunk};

		npy::SaveArrayAsNumpy("rowfft.npy", false, leshape2.size(), leshape2.data(), v2);

		int ijt = 0;
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
		cout << "n_coherent = " << n_coherent << endl;
		
		// !3

		//4.
		complex<float>* pcarrCD_Out = (complex<float>*)malloc(sizeof(complex<float>) * LEnChunk);
		complex<float>* pcarrTemp = (complex<float>*)malloc(sizeof(complex<float>) * (LEnChunk/ N_p) * N_p);
		float* parr_fdmt_inp = (float*)malloc(sizeof(float) * (LEnChunk / N_p) * N_p);
		float* parr_fdmt_out = (float*)malloc(sizeof(float) * (LEnChunk / N_p) * N_p);
		for (int iouter_d = 0; iouter_d < n_coherent; ++iouter_d)
		//for (int iouter_d = 31; iouter_d < 32; ++iouter_d)
		{
			cout <<	"coherent iteration " << iouter_d << endl;
			
			long double valcur_coherent_d = ((long double)iouter_d) * ((long double)VAlD_max / ((long double)n_coherent));
			cout << "cur_coherent_d = " << valcur_coherent_d << endl;

			// !!!
			std::vector<complex<float>> v6(pffted_rowsignal, pffted_rowsignal + LEnChunk);

			std::array<long unsigned, 1> leshape6{ LEnChunk};

			npy::SaveArrayAsNumpy("ffted.npy", false, leshape6.size(), leshape6.data(), v6);
			int ijt1 = 0;

			fncCoherentDedispersion(pcarrCD_Out, pffted_rowsignal
				, LEnChunk, DISPERSION_CONSTANT * valcur_coherent_d, VAlFmin, VAlFmax);

			
			// !!!
			std::vector<complex<float>> v3(pcarrCD_Out, pcarrCD_Out + LEnChunk);

			std::array<long unsigned, 1> leshape3{ LEnChunk};

			npy::SaveArrayAsNumpy("CoherentDedispersion.npy", false, leshape3.size(), leshape3.data(), v3);

			int ijt6 = 0;

			fncSTFT(pcarrTemp, pcarrCD_Out, LEnChunk, N_p);

			// !!!
			std::vector<complex<float>> v4(pcarrTemp, pcarrTemp + LEnChunk);

			std::array<long unsigned, 2> leshape4{N_p, LEnChunk / N_p};

			npy::SaveArrayAsNumpy("STFT.npy", false, leshape4.size(), leshape4.data(), v4);

			int ijt2 = 0;

			
			float sum = 0.;
			int len = (LEnChunk / N_p) * N_p;
			for (int j = 0; j < len ; ++j)
			{
				float temp = pcarrTemp[j].real() * pcarrTemp[j].real() + pcarrTemp[j].imag() * pcarrTemp[j].imag();
				
				sum += temp;
				parr_fdmt_inp[j] = temp;
			}

			// !!!
			std::vector<complex<float>> v5(parr_fdmt_inp, parr_fdmt_inp + len);

			std::array<long unsigned, 2> leshape5{N_p, LEnChunk / N_p};

			npy::SaveArrayAsNumpy("inp0.npy", false, leshape5.size(), leshape5.data(), v5);

			int ijt3 = 0;

			float val_mean = sum / ((float)len);
			float valStdDev = fnsStdDev(parr_fdmt_inp, val_mean, len);

			for (int j = 0; j < len; ++j)
			{
				parr_fdmt_inp[j] = (parr_fdmt_inp[j] - val_mean) / (0.25 * valStdDev);
			}
			


			// !!!
			std::vector<float> v1(parr_fdmt_inp, parr_fdmt_inp + N_p * (LEnChunk / N_p));

			std::array<long unsigned, 2> leshape1{N_p, LEnChunk / N_p};

			npy::SaveArrayAsNumpy("inp.npy", false, leshape1.size(), leshape1.data(), v1);

			int ij = 0;
			
			fncFdmt_cpuF_v0(parr_fdmt_inp, N_p, LEnChunk/ N_p
				, VAlFmin, VAlFmax, IMaxDT, parr_fdmt_out);

			// !!!
			std::vector<float> v0(parr_fdmt_out, parr_fdmt_out + N_p * (LEnChunk / N_p));

			std::array<long unsigned, 2> leshape0{N_p, LEnChunk / N_p};

			npy::SaveArrayAsNumpy("new.npy", false, leshape0.size(), leshape0.data(), v0);

			int ii = 0;
			
			float* pmaxElement0 = std::max_element(parr_fdmt_out, parr_fdmt_out + N_p * (LEnChunk / N_p));
			float maxElem0 = *pmaxElement0;

			float val_V = 0., val_mean1 = 0.;
			fncDisp(parr_fdmt_inp, len, val_mean1,val_V);
			float* p = parr_fdmt_out;
			float* pn = piarrImNormalize;
			for (int i = 0; i < len; ++i)
			{
				*p = (*p) / sqrt(((*pn) * val_V + 0.000001));
				++p;
				++pn;
			}
			
			float* pmaxElement1 = std::max_element(parr_fdmt_out, parr_fdmt_out + N_p * (LEnChunk / N_p));
			float maxElem1 = *pmaxElement1;
			if (maxElem1 > valSigmaBound)
			{
				valSigmaBound = maxElem0;
				coherent_d = valcur_coherent_d;
				cout <<"!!!!!!! achieved score with " << valSigmaBound <<"!!!!!!!"<< endl;
				if (nullptr != poutImage)
				{
					memcpy(poutImage, parr_fdmt_out, len * sizeof(float));
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

	// !!!
	std::vector<long double> v3(pcarrf, pcarrf + LEnChunk);

	std::array<long unsigned, 1> leshape3{ LEnChunk};

	npy::SaveArrayAsNumpy("f.npy", false, leshape3.size(), leshape3.data(), v3);


	complex<long double>* pcarrH = (complex<long double>*)malloc(LEnChunk * sizeof(complex<long double>));
	
	complex <float> cmp_temp (0.,   -2. * M_PI * VAl_practicalD) ;
	//float val_fcur = 0.;
	//H = np.e * *(-(2 * np.pi * complex(0, 1) * practicalD / (f_min + f) + 2 * np.pi * complex(0, 1) * practicalD * f / (f_max * *2)))
	for (int i = 0; i < LEnChunk; ++i)
	{
		if (739823 == i)
		{
			int hghghgh = 0;
		}
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

	// !!!
	std::vector<complex<long double>> v1(pcarrH, pcarrH + LEnChunk);

	std::array<long unsigned, 1> leshape1{ LEnChunk};

	npy::SaveArrayAsNumpy("H.npy", false, leshape1.size(), leshape1.data(), v1);

	
	complex<float>* pcarrTemp = (complex<float>*)malloc(LEnChunk * sizeof(complex<float>));
	for (int i = 0; i < LEnChunk; ++i)
	{
		pcarrTemp[i] = (complex<float>)pcarrH[i] * pcarrffted_rowsignal[i];
	}
	free(pcarrH);
	free(pcarrf);


	// !!!
	std::vector<complex<float>> v0(pcarrTemp, pcarrTemp + LEnChunk);

	std::array<long unsigned, 1> leshape0{ LEnChunk};

	npy::SaveArrayAsNumpy("xx.npy", false, leshape0.size(), leshape0.data(), v0);

	int ii = 0;

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