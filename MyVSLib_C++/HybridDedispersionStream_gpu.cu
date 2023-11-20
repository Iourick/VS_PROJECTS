#include "HybridDedispersionStream_gpu.cuh"
#include "FdmtCpuT_omp.h"
#include "StreamParams.h"
#include "Constants.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <array>
#include <vector>
#include "npy.hpp"
#include <cufft.h>
#include <thrust/complex.h>
#include <complex>





#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


using namespace std;

int fncHybridDedispersionStream( int* piarrNumSuccessfulChunks, float* parrCoherent_d
	, int& quantOfSuccessfulChunks, CStreamParams* pStreamPars)
{
	const int NumChunks = ((pStreamPars->m_numEnd - pStreamPars->m_numBegin) + pStreamPars->m_lenChunk - 1) / pStreamPars->m_lenChunk;
	std::complex<float>* pRawSignalCur = (std::complex<float>*)malloc(sizeof(complex<float>*) * pStreamPars->m_lenChunk);// new complex<float>[pStreamPars->m_lenChunk];
	quantOfSuccessfulChunks = 0;

	// remains not readed elements
	int iremains = pStreamPars->m_lenarr;
	float val_coherent_d;
	for (int i = 0; i < NumChunks; ++i)
	{
		int length = (iremains < pStreamPars->m_lenChunk) ? iremains : pStreamPars->m_lenChunk;

		fread(pRawSignalCur, sizeof(std::complex<float>), length, pStreamPars->m_stream);
		
		if (fncChunkHybridDedispersion(pRawSignalCur, length, pStreamPars->m_n_p
			, pStreamPars->m_D_max, pStreamPars->m_f_min, pStreamPars->m_f_max, pStreamPars->m_SigmaBound, val_coherent_d))

		{
			piarrNumSuccessfulChunks[quantOfSuccessfulChunks] = i;
			parrCoherent_d[quantOfSuccessfulChunks] = val_coherent_d;
			++quantOfSuccessfulChunks;
		}
		++(pStreamPars->m_numCurChunk);
		iremains -= pStreamPars->m_lenChunk;
	}
	
	
	free(pRawSignalCur);
	return 0;
}
//-------------------------------------------------------------------------------------------------

bool fncChunkHybridDedispersion(std::complex<float>* pRawSignalCur
	, const unsigned int LEnChunk, const unsigned int N_p
	, const float VAlD_max, const float VAlFmin, const float VAlFmax, float& valSigmaBound_, float& coherent_d)
{
	//bool bres = false;
	//coherent_d = -1.;
	//float valSigmaBound = valSigmaBound_;
	//// 1. create FFT
	//complex<float>* pffted_rowsignal = (complex<float>*)malloc(sizeof(complex<float>) * LEnChunk);
	//fftwf_complex* fftw_out = reinterpret_cast<fftwf_complex*>(pffted_rowsignal);
	//fftwf_complex* fftw_in = reinterpret_cast<fftwf_complex*>(pRawSignalCur);
	//// Create the FFT plan
	//fftwf_plan plan = fftwf_plan_dft_1d(LEnChunk, fftw_in, fftw_out, FFTW_FORWARD, FFTW_ESTIMATE);

	//// Execute the FFT
	//fftwf_execute(plan);
	//fftwf_destroy_plan(plan);

	//// !1

	//	// 2. create fdmt ones
	//float* parr_fdmt_ones = (float*)malloc(LEnChunk * sizeof(float));
	//for (int i = 0; i < LEnChunk; ++i)
	//{
	//	parr_fdmt_ones[i] = 1.;
	//}
	//int IMaxDT = N_p;
	//float* parrImNormalize = (float*)malloc(N_p * (LEnChunk / N_p) * sizeof(float));

	////fncFdmt_cpuF_v0(parr_fdmt_ones, N_p, LEnChunk / N_p
	////	, VAlFmin, VAlFmax, IMaxDT, parrImNormalize);


	//fncFdmt_cpuT_v0(parr_fdmt_ones, N_p, LEnChunk / N_p
	//	, VAlFmin, VAlFmax, IMaxDT, parrImNormalize);


	//free(parr_fdmt_ones);
	//// !2

	//// 3.		
	//float valConversionConst = DISPERSION_CONSTANT * (1. / (VAlFmin * VAlFmin) - 1. / (VAlFmax * VAlFmax)) * (VAlFmax - VAlFmin);
	//float valN_d = VAlD_max * valConversionConst;
	//int n_coherent = int(ceil(valN_d / (N_p * N_p)));
	//cout << "n_coherent = " << n_coherent << endl;

	//// !3

	////4.
	//complex<float>* pcarrCD_Out = (complex<float>*)malloc(sizeof(complex<float>) * LEnChunk);
	//complex<float>* pcarrTemp = (complex<float>*)malloc(sizeof(complex<float>) * (LEnChunk / N_p) * N_p);
	//float* parr_fdmt_inp = (float*)malloc(sizeof(float) * (LEnChunk / N_p) * N_p);
	//float* parr_fdmt_out = (float*)malloc(sizeof(float) * (LEnChunk / N_p) * N_p);

	//for (int iouter_d = 31; iouter_d < 32; ++iouter_d)
	//	//for (int iouter_d = 0; iouter_d < n_coherent; ++iouter_d)
	//{
	//	cout << "coherent iteration " << iouter_d << endl;

	//	long double valcur_coherent_d = ((long double)iouter_d) * ((long double)VAlD_max / ((long double)n_coherent));
	//	cout << "cur_coherent_d = " << valcur_coherent_d << endl;

	//	createOutputFDMT(parr_fdmt_out, pffted_rowsignal, pcarrCD_Out, pcarrTemp
	//		, LEnChunk, N_p, parr_fdmt_inp, IMaxDT
	//		, DISPERSION_CONSTANT * valcur_coherent_d, VAlD_max, VAlFmin, VAlFmax);
	//	int len = (LEnChunk / N_p) * N_p;

	//	float maxSig = -1.;
	//	int iargmax = -1;
	//	fncMaxSignalDetection(parr_fdmt_out, parrImNormalize, N_p, (LEnChunk / N_p)
	//		, &maxSig, &iargmax);

	//	if (maxSig > valSigmaBound)
	//	{
	//		valSigmaBound = maxSig;
	//		coherent_d = valcur_coherent_d;
	//		cout << "!!!!!!! achieved score with " << valSigmaBound << "!!!!!!!" << endl;
	//		bres = true;
	//		if (nullptr != poutImage)
	//		{
	//			memcpy(poutImage, parr_fdmt_out, len * sizeof(float));
	//		}
	//		std::cout << "SNR = " << maxSig << endl;
	//		std::cout << "NUM ARGMAX = " << iargmax << endl;
	//		std::cout << "ROW ARGMAX = " << (int)(iargmax / (LEnChunk / N_p)) << endl;
	//		std::cout << "COLUMN ARGMAX = " << (int)(iargmax % (LEnChunk / N_p)) << endl;

	//	}
	//}
	//free(pcarrTemp);
	//free(pcarrCD_Out);
	//free(parr_fdmt_inp);
	//free(parr_fdmt_out);
	//free(parrImNormalize);
	//return bres;
	return true;
}
