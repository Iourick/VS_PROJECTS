#include "HybridDedispersionStream_gpu.cuh"
#include "fdmtU_cu.cuh"
#include "StreamParams.h"
#include "Constants.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <array>
#include <vector>
#include "npy.hpp"
//#include <cufft.h>
//#include <thrust/complex.h>
//#include <complex>

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


using namespace std;

int fncHybridDedispersionStream( int* piarrNumSuccessfulChunks, float* parrCoherent_d
	, int& quantOfSuccessfulChunks, CStreamParams* pStreamPars)
{
	// 1. memory allocation for curreny chunk
	const int NumChunks = ((pStreamPars->m_numEnd - pStreamPars->m_numBegin) + pStreamPars->m_lenChunk - 1) / pStreamPars->m_lenChunk;
	cufftComplex* pcmparrRawSignalCur = NULL;
	cudaMallocManaged(&pcmparrRawSignalCur, pStreamPars->m_lenChunk * sizeof(cufftComplex));
	// 1!

	// 2. memory allocation for fdmt_ones on GPU
	float* d_arrfdmt_norm = 0;
	cudaMalloc(&d_arrfdmt_norm, pStreamPars->m_lenChunk * sizeof(cufftComplex));
	// 2!

	// 3.memory allocation for auxillary buffer for fdmt
	const int  IDeltaT = calc_IDeltaT(pStreamPars->m_n_p, pStreamPars->m_f_min, pStreamPars->m_f_max, pStreamPars->m_IMaxDT);
	size_t szBuff_fdmt = calcSizeAuxBuff_fdmt(pStreamPars->m_n_p, pStreamPars->m_lenChunk / pStreamPars->m_n_p
		, pStreamPars->m_f_min, pStreamPars->m_f_max, pStreamPars->m_IMaxDT);
	void* pAuxBuff_fdmt = 0;
	cudaMalloc(&pAuxBuff_fdmt, szBuff_fdmt);
	// 3!

	// 4. memory allocation for the rest of auxillary arrays	
	size_t szBuff_the_rest = 3 * (pStreamPars->m_lenChunk) * sizeof(cufftComplex) + 2 * (pStreamPars->m_lenChunk) * sizeof(float);
	void* pAuxBuff_the_rest = 0;
	cudaMalloc(&pAuxBuff_the_rest, szBuff_the_rest);
	// 4!
	
	// 5. calculation fdmt ones
	fncFdmtU_cu(
		  nullptr      // on-device input image
		, pAuxBuff_fdmt
		, pStreamPars->m_n_p
		, pStreamPars->m_lenChunk / pStreamPars->m_n_p // dimensions of input image 	
		, IDeltaT
		, pStreamPars->m_f_min
		, pStreamPars->m_f_max
		, pStreamPars->m_IMaxDT
		, d_arrfdmt_norm	// OUTPUT image, dim = IDeltaT x IImgcols
		, true
	);
	
	quantOfSuccessfulChunks = 0;

	// remains not readed elements
	int iremains = pStreamPars->m_lenarr;
	float val_coherent_d;
	for (int i = 0; i < NumChunks; ++i)
	{
		int length = (iremains < pStreamPars->m_lenChunk) ? iremains : pStreamPars->m_lenChunk;

		fread(pcmparrRawSignalCur, sizeof(cufftComplex), length, pStreamPars->m_stream);
		
		if (fncChunkHybridDedispersion_gpu(pcmparrRawSignalCur, length, pStreamPars->m_n_p
			, pStreamPars->m_D_max, pStreamPars->m_IMaxDT, pStreamPars->m_f_min, pStreamPars->m_f_max
			, pStreamPars->m_SigmaBound, val_coherent_d, pAuxBuff_fdmt,  pAuxBuff_the_rest,  d_arrfdmt_norm))

		{
			piarrNumSuccessfulChunks[quantOfSuccessfulChunks] = i;
			parrCoherent_d[quantOfSuccessfulChunks] = val_coherent_d;
			++quantOfSuccessfulChunks;
		}
		++(pStreamPars->m_numCurChunk);
		iremains -= pStreamPars->m_lenChunk;
	}
	
	cudaFree(pcmparrRawSignalCur);
	cudaFree(d_arrfdmt_norm);
	cudaFree(pAuxBuff_fdmt);
	cudaFree(pAuxBuff_the_rest);
	return 0;
}
//-------------------------------------------------------------------------------------------------
bool fncChunkHybridDedispersion_gpu(cufftComplex* pcmparrRawSignalCur
	, const unsigned int LEnChunk, const unsigned int N_p
	, const float VAlD_max, const int IMaxDT, const float VAlFmin, const float VAlFmax, float& valSigmaBound_, float& coherent_d
	,void* pAuxBuff_fdmt, void* pAuxBuff_the_rest, float * d_arrfdmt_norm)
{
	// installation of pointers	for pAuxBuff_the_rest
	cufftComplex* pffted_rowsignal = (cufftComplex*)pAuxBuff_the_rest;  //1
	size_t sz0 = LEnChunk * sizeof(cufftComplex);
	cufftComplex* pcarrTemp = (cufftComplex*)((char*)pAuxBuff_the_rest + sz0); //2
	size_t sz1 = sz0 + LEnChunk * sizeof(cufftComplex);
	cufftComplex* pcarrCD_Out = (cufftComplex*)((char*)pAuxBuff_the_rest + sz1); //3
	size_t sz2 = sz1 + LEnChunk * sizeof(cufftComplex);

	float* parr_fdmt_inp = (float*)((char*)pAuxBuff_the_rest + sz2); //4
	size_t sz3 = sz2 + sizeof(float) * LEnChunk;
	float* parr_fdmt_out = (float*)((char*)pAuxBuff_the_rest + sz3); //5

	bool bres = false;
	coherent_d = -1.;
	float valSigmaBound = valSigmaBound_;
	// 1. create FFT
	
	//// Create the FFT plan
	//fftwf_plan plan = fftwf_plan_dft_1d(LEnChunk, fftw_in, fftw_out, FFTW_FORWARD, FFTW_ESTIMATE);

	//// Execute the FFT
	//fftwf_execute(plan);
	//fftwf_destroy_plan(plan);

	// !1

	// !2

	// 3.		
	float valConversionConst = DISPERSION_CONSTANT * (1. / (VAlFmin * VAlFmin) - 1. / (VAlFmax * VAlFmax)) * (VAlFmax - VAlFmin);
	float valN_d = VAlD_max * valConversionConst;
	int n_coherent = int(ceil(valN_d / (N_p * N_p)));
	cout << "n_coherent = " << n_coherent << endl;

	// !3

	//4.
	

	for (int iouter_d = 31; iouter_d < 32; ++iouter_d)
		//for (int iouter_d = 0; iouter_d < n_coherent; ++iouter_d)
	{
		cout << "coherent iteration " << iouter_d << endl;

		long double valcur_coherent_d = ((long double)iouter_d) * ((long double)VAlD_max / ((long double)n_coherent));
		cout << "cur_coherent_d = " << valcur_coherent_d << endl;

		/*createOutputFDMT(parr_fdmt_out, pffted_rowsignal, pcarrCD_Out, pcarrTemp
			, LEnChunk, N_p, parr_fdmt_inp, IMaxDT
			, DISPERSION_CONSTANT * valcur_coherent_d, VAlD_max, VAlFmin, VAlFmax);*/
		int len = (LEnChunk / N_p) * N_p;

		float maxSig = -1.;
		int iargmax = -1;
		/*fncMaxSignalDetection(parr_fdmt_out, parrImNormalize, N_p, (LEnChunk / N_p)
			, &maxSig, &iargmax);*/

		if (maxSig > valSigmaBound)
		{
			valSigmaBound = maxSig;
			coherent_d = valcur_coherent_d;
			cout << "!!!!!!! achieved score with " << valSigmaBound << "!!!!!!!" << endl;
			bres = true;
			/*if (nullptr != poutImage)
			{
				memcpy(poutImage, parr_fdmt_out, len * sizeof(float));
			}*/
			std::cout << "SNR = " << maxSig << endl;
			std::cout << "NUM ARGMAX = " << iargmax << endl;
			std::cout << "ROW ARGMAX = " << (int)(iargmax / (LEnChunk / N_p)) << endl;
			std::cout << "COLUMN ARGMAX = " << (int)(iargmax % (LEnChunk / N_p)) << endl;

		}
	}
	free(pcarrTemp);
	free(pcarrCD_Out);
	free(parr_fdmt_inp);
	free(parr_fdmt_out);
	
	return bres;
	return true;
}
