#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Block.cuh"
#include "OutChunk.h"
#include <vector>
#include "OutChunkHeader.h"

#include "Constants.h"
#include "fdmtU_cu.cuh"
#include "Constants.h"
#include <chrono>

#include "helper_functions.h"
#include "helper_cuda.h"
#include <math_functions.h>
#include "aux_kernels.cuh"
#include "Detection.cuh"
#include "Cleaning.cuh"
#include <complex>
#include "yr_cart.h"




#ifdef _WIN32 // Windows

#include <Windows.h>

    void emitSound(int frequency, int duration) {
        Beep(frequency, duration);
    }

#else // Linux

#include <cmath>
#include <alsa/asoundlib.h>

    void emitSound(int frequency, int duration) {
        int rate = 44100; // Sampling rate
        snd_pcm_t* handle;
        snd_pcm_open(&handle, "default", SND_PCM_STREAM_PLAYBACK, 0);
        snd_pcm_set_params(handle, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED, 1, rate, 1, 500000);

        short buf[rate * duration];

        for (int i = 0; i < rate * duration; i++) {
            int sample = 32760 * sin(2 * M_PI * frequency * i / rate);
            buf[i] = sample;
        }

        snd_pcm_writei(handle, buf, rate * duration);
        snd_pcm_close(handle);
    }

#endif   


// timing variables:
  // fdmt time
long int iFdmt_time = 0;
// read && transform data time
long int iReadTransform_time = 0;
// fft time
long int iFFT_time = 0;
// detection time
long int iMeanDisp_time = 0;
// detection time
long int iNormalize_time = 0;
// total time
long int iTotal_time = 0;

#define TILE_DIM 32

CBlock::~CBlock()
{
	if (m_pvctSuccessHeaders)
	{
		delete m_pvctSuccessHeaders;
	}
}
CBlock::CBlock()
{
	m_Fmin = 0;
	m_Fmax = 0;
	m_npol = 0;
	m_nblocksize = 0;
	m_nchan = 0;	
	m_lenChunk = 0;
	m_len_sft = 0;	
	m_pvctSuccessHeaders = new std::vector<COutChunkHeader>();
	m_block_id = -1;
	m_nbits = 0;
	m_enchannelOrder = STRAIGHT;
	m_d_max = 0.;
	m_sigma_bound = 10.;
	m_length_sum_wnd = 10;
}
//-----------------------------------------------------------

CBlock::CBlock(const  CBlock& R)
{
	m_Fmin = R.m_Fmin;
	m_Fmax = R.m_Fmax;
	m_npol = R.m_npol;
	m_nblocksize = R.m_nblocksize;
	m_nchan = R.m_nchan;	
	m_lenChunk = R.m_lenChunk;
	m_len_sft = R.m_len_sft;
	if (m_pvctSuccessHeaders)
	{
		m_pvctSuccessHeaders = R.m_pvctSuccessHeaders;
	}
	m_block_id = R.m_block_id;
	m_nbits = R.m_nbits;
	m_enchannelOrder = R.m_enchannelOrder;
	m_d_max = R.m_d_max;
	m_sigma_bound = R.m_sigma_bound;
	m_length_sum_wnd = R.m_length_sum_wnd;
}
//-------------------------------------------------------------------

CBlock& CBlock::operator=(const CBlock& R)
{
	if (this == &R)
	{
		return *this;
	}

	m_Fmin = R.m_Fmin;
	m_Fmax = R.m_Fmax;
	m_npol = R.m_npol;
	m_nblocksize = R.m_nblocksize;
	m_nchan = R.m_nchan;	
	m_lenChunk = R.m_lenChunk;
	m_len_sft = R.m_len_sft;
	if (m_pvctSuccessHeaders)
	{
		m_pvctSuccessHeaders = R.m_pvctSuccessHeaders;
	}
	m_block_id = R.m_block_id;
	m_nbits = R.m_nbits;
	m_enchannelOrder = R.m_enchannelOrder;
	m_d_max = R.m_d_max;
	m_sigma_bound = R.m_sigma_bound;
	m_length_sum_wnd = R.m_length_sum_wnd;
	return *this;
}
//------------------------------------------------------------------
CBlock::CBlock(
	const float Fmin
	, const float Fmax
	, const int npol
	, const int nblocksize
	, const int nchan		
	, const unsigned int lenChunk
	, const unsigned int len_sft
	, const int bulk_id
	, const int nbits
	, const bool bOrderStraight
	, const double d_max
	, const float sigma_bound
	, const int length_sum_wnd
)
{
	m_Fmin = Fmin;

	m_Fmax = Fmax;

	m_npol = npol;

	m_nchan = nchan;

	m_nblocksize = nblocksize;	

	m_lenChunk = lenChunk;

	m_len_sft = len_sft;
	
	m_pvctSuccessHeaders = new std::vector<COutChunkHeader>();

	m_block_id = bulk_id;

	m_nbits = nbits;

	if (bOrderStraight)
	{
		m_enchannelOrder = STRAIGHT;
	}
	else
	{
		m_enchannelOrder = INVERTED;
	}

	m_d_max = d_max;

	m_sigma_bound = sigma_bound;

	m_length_sum_wnd = length_sum_wnd;
}
//-----------------------------------------
int CBlock::process(FILE * rb_file,int* pquantSuccessChunks)
{
	cout << "Block ID = " << m_block_id  << endl;	
	
	// total number of downloding bytes to each chunk:
	const long long QUantTotalChunkBytes = m_lenChunk / 8 * m_nchan * m_npol * m_nbits;
	// total number of downloding bytes to each channel:
	const long long QUantTotalChannelBytes =  m_nblocksize * m_nbits / 8/ m_nchan;
	// total number of downloding words of each chunk:
	const long long QUantChunkWords = m_lenChunk * m_nchan * m_npol;
	// total number of downloding words of each channel:
	const long long QUantChannelWords = m_lenChunk *  m_npol;
	// total number of downloading complex numbers of chunk:
	const long long QUantChunkComplexNumbers = QUantChunkWords / 2;
	// total number of downloading complex numbers of channel:
	const long long QUantChannelComplexNumbers = QUantChannelWords / 2;

	const int NumChunks = (m_nblocksize + QUantTotalChunkBytes - 1) / QUantTotalChunkBytes;

	// 1. memory allocation for input array, got from input file, with input type of data
	char* d_parrInput = NULL;
	checkCudaErrors(cudaMallocManaged((void**)&d_parrInput, QUantTotalChunkBytes));
	// 2. memory allocation for current chunk	
	cufftComplex* pcmparrRawSignalCur = NULL;
	checkCudaErrors(cudaMallocManaged((void**)&pcmparrRawSignalCur, QUantChunkComplexNumbers * sizeof(cufftComplex)));
	// 2!

	// 3. memory allocation for fdmt_ones on GPU  ????
	fdmt_type_* d_arrfdmt_norm = 0;
	checkCudaErrors(cudaMalloc((void**)&d_arrfdmt_norm, m_lenChunk  * sizeof(fdmt_type_)));
	// 3!

	// 4.memory allocation for auxillary buffer for fdmt
	const int N_p = m_len_sft * m_nchan;
	unsigned int IMaxDT = m_len_sft * m_nchan;
	const int  IDeltaT = calc_IDeltaT(N_p, m_Fmin, m_Fmax, IMaxDT);
	size_t szBuff_fdmt = calcSizeAuxBuff_fdmt(N_p, m_lenChunk / N_p
		, m_Fmin, m_Fmax, IMaxDT);
	void* pAuxBuff_fdmt = 0;
	checkCudaErrors(cudaMalloc(&pAuxBuff_fdmt, szBuff_fdmt));	
	// 4!

	// 5. memory allocation for the 4 auxillary cufftComplex  arrays on GPU	
	//cufftComplex* pffted_rowsignal = NULL; //1	
	cufftComplex* pcarrTemp = NULL; //2	
	cufftComplex* pcarrCD_Out = NULL;//3
	cufftComplex* pcarrBuff = NULL;//3
	

	checkCudaErrors(cudaMalloc((void**)&pcarrTemp, QUantChunkComplexNumbers * sizeof(cufftComplex)));

	checkCudaErrors(cudaMalloc((void**)&pcarrCD_Out, QUantChunkComplexNumbers * sizeof(cufftComplex)));

	checkCudaErrors(cudaMalloc((void**)&pcarrBuff, QUantChunkComplexNumbers * sizeof(cufftComplex)));
	// !5

	// 5. memory allocation for the 2 auxillary float  arrays on GPU	
	float* pAuxBuff_flt = NULL;
	checkCudaErrors(cudaMalloc((void**)&pAuxBuff_flt, 2 * m_lenChunk * sizeof(float)));
	
	// 5!

	// 6. calculation fdmt ones
	fncFdmtU_cu(
		nullptr      // on-device input image
		, pAuxBuff_fdmt
		, N_p
		, m_lenChunk/ N_p // dimensions of input image 	
		, IDeltaT
		, m_Fmin
		, m_Fmax
		, IMaxDT
		, d_arrfdmt_norm	// OUTPUT image, dim = IDeltaT x IImgcols
		, true
	);
	/*float* arrfdmt_norm = (float*)malloc(m_lenChunk * sizeof(float));
	cudaMemcpy(arrfdmt_norm, d_arrfdmt_norm, m_lenChunk * sizeof(float), cudaMemcpyDeviceToHost);
	float valmax1 = -0., valmin1 = 0.;
	unsigned int iargmax1 = -1, iargmin1 = -1;
	findMaxMinOfArray(arrfdmt_norm, m_lenChunk, &valmax1, &valmin1
		, &iargmax1, &iargmin1);*/

	// !6


	// 7. remains of not readed elements in block
	long long iremainedBytes = m_nblocksize;
	float val_coherent_d;
	// !7

	//// 8.  Array to store cuFFT plans for different array sizes

	cufftHandle plan0 = NULL;
	cufftCreate(&plan0);	
	checkCudaErrors(cufftPlan1d(&plan0, m_lenChunk, CUFFT_C2C, m_nchan * m_npol/2));

	cufftHandle plan1 = NULL;
	cufftCreate(&plan1);	
	checkCudaErrors(cufftPlan1d(&plan1, m_len_sft, CUFFT_C2C, m_lenChunk *m_nchan * m_npol / 2/ m_len_sft));	
	

	// 9. main loop
	auto start = std::chrono::high_resolution_clock::now();

	float valSNR = -1;
	int argmaxRow = -1;
	int argmaxCol = -1;
	float coherentDedisp = -1.;
	*pquantSuccessChunks = 0;
	structOutDetection* pstructOut = NULL;
	checkCudaErrors(cudaMallocManaged((void**)&pstructOut, sizeof(structOutDetection)));
	for (int i = 0; i < NumChunks; ++i)
	{
		long long quantDownloadingBytes = (iremainedBytes < QUantTotalChunkBytes) ? iremainedBytes : QUantTotalChunkBytes;
		auto start = std::chrono::high_resolution_clock::now();

		size_t sz = downloadChunk(rb_file, d_parrInput, quantDownloadingBytes);

		/*std::vector<inp_type_ > data0(m_lenChunk, 0);
		cudaMemcpy(data0.data(), d_parrInput, m_lenChunk * sizeof(inp_type_ ),
			cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();*/

		const dim3 blockSize = dim3(1024, 1, 1);
		const dim3 gridSize = dim3((m_lenChunk + blockSize.x - 1) / blockSize.x, m_nchan, 1);
		unpackInput << < gridSize, blockSize >> > (pcmparrRawSignalCur,(inp_type_ *)d_parrInput, m_lenChunk, m_nchan, m_npol);
		cudaDeviceSynchronize();

		/* std::vector<std::complex<float>> data(m_lenChunk, 0);
		cudaMemcpy(data.data(), pcmparrRawSignalCur, m_lenChunk * sizeof(cufftComplex),
			cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();*/
		//std::array<long unsigned, 1> leshape127{ LEnChunk };
		//npy::SaveArrayAsNumpy("ffted.npy", false, leshape127.size(), leshape127.data(), data);
		
		if (fncChunkProcessing_gpu(pcmparrRawSignalCur
			, pAuxBuff_fdmt			
			, pcarrTemp
			, pcarrCD_Out
			, pcarrBuff
			, pAuxBuff_flt, d_arrfdmt_norm
			, IDeltaT, plan0, plan1
			, pstructOut
			, &coherentDedisp))

		{			
				COutChunkHeader head(N_p
					, m_lenChunk / N_p
					, (*pstructOut).icol
					, (*pstructOut).icol
					, (*pstructOut).iwidth
					, (*pstructOut).snr					
					, coherentDedisp
					, m_block_id
					, i );
				m_pvctSuccessHeaders->push_back(head);
				++(*pquantSuccessChunks);
			
			
		}
		
	}
	// rewind to the beginning of the data block
	fseek(rb_file,  -QUantTotalChannelBytes, SEEK_CUR);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	//std::cout << "/*****************************************************/ " << std::endl;
	//std::cout << "/*****************************************************/ " << std::endl;
	//std::cout << "/*****************************************************/ " << std::endl;
	//iTotal_time = duration.count();
	//std::cout << "Total time:                   " << iTotal_time << " microseconds" << std::endl;
	//std::cout << "FDMT time:                    " << iFdmt_time << " microseconds" << std::endl;
	//std::cout << "Read and data transform time: " << iReadTransform_time << " microseconds" << std::endl;
	//std::cout << "FFT time:                     " << iFFT_time << " microseconds" << std::endl;
	//std::cout << "Mean && disp time:            " << iMeanDisp_time << " microseconds" << std::endl;
	//std::cout << "Normalization time:           " << iNormalize_time << " microseconds" << std::endl;

	//std::cout << "/*****************************************************/ " << std::endl;
	//std::cout << "/*****************************************************/ " << std::endl;
	//std::cout << "/*****************************************************/ " << std::endl;



	cudaFree(pcmparrRawSignalCur);
	cudaFree(d_arrfdmt_norm);
	cudaFree(pAuxBuff_fdmt);	
	cudaFree(pcarrTemp); //2	
	cudaFree(pcarrCD_Out);//3
	cudaFree(pcarrBuff);//3
	cudaFree(pAuxBuff_flt);
	
	cufftDestroy(plan0);
	cufftDestroy(plan1);
	//
	cudaFree(d_parrInput);
	cudaFree(pstructOut);
	
	
	return 0;
}

//------------------------------------------
size_t  CBlock::downloadChunk(FILE* rb_file,char* d_parrInput, const long long QUantDownloadingBytes)
{	
	long long quantDownloadingBytesPerChannel = QUantDownloadingBytes / m_nchan;
	long long quantTotalBytesPerChannel = m_nblocksize / m_nchan;

	//
	// total number of downloding bytes to each chunk:
	const long long QUantChunkBytes = m_lenChunk * m_nchan * m_npol * m_nbits / 8;
	// total number of downloding bytes to each channel:
	const long long QUantChannelBytes = QUantChunkBytes / m_nchan;
	// total number of downloding words of each chunk:
	const long long QUantChunkWords = m_lenChunk * m_nchan * m_npol;
	// total number of downloding words of each channel:
	const long long QUantChannelWords = m_lenChunk * m_npol;
	// total number of downloading complex numbers of chunk:
	const long long QUantChunkComplexNumbers = QUantChunkWords / 2;
	// total number of downloading complex numbers of channel:
	const long long QUantChannelComplexNumbers = QUantChannelWords / 2;

	//
	
	char* p = (m_enchannelOrder == STRAIGHT) ? d_parrInput : d_parrInput + (m_nchan - 1) * quantDownloadingBytesPerChannel;
	size_t sz_rez = 0;
	for (int i = 0; i < m_nchan; ++i)
	{
		sz_rez += fread(p, sizeof(char), quantDownloadingBytesPerChannel, rb_file);
		if (m_enchannelOrder == STRAIGHT)
		{
			p += quantDownloadingBytesPerChannel;
		}
		else
		{
			p -= quantDownloadingBytesPerChannel;
		}

		fseek(rb_file, quantTotalBytesPerChannel - quantDownloadingBytesPerChannel, SEEK_CUR);

	}
	
	fseek(rb_file, -quantTotalBytesPerChannel * (m_nchan - 1) /*+ quantDownloadingBytesPerChannel*/, SEEK_CUR);
	
	return sz_rez;
}
//------------------------------------------
size_t  downloadChunk_(char* d_parrInput, const long long QUantDownloadingBytes , const int nchan
	,const long long nblocksize, const EN_channelOrder enchannelOrder, FILE* rb_file)
{
	long long quantDownloadingBytesPerChannel = QUantDownloadingBytes / nchan;
	long long quantTotalBytesPerChannel = nblocksize / nchan;
	long long end_position = ftell(rb_file) + quantDownloadingBytesPerChannel;
	char* p = (enchannelOrder == STRAIGHT) ? d_parrInput : d_parrInput + (nchan - 1) * quantDownloadingBytesPerChannel;
	size_t sz_rez = 0;
	
	for (int i = 0; i < nchan; ++i)
	{
		sz_rez += fread(p, sizeof(char), quantDownloadingBytesPerChannel, rb_file);
		
		if (enchannelOrder == STRAIGHT)
		{
			p += quantDownloadingBytesPerChannel;
		}
		else
		{
			p -= quantDownloadingBytesPerChannel;
		}

		fseek(rb_file, quantTotalBytesPerChannel - quantDownloadingBytesPerChannel, SEEK_CUR);		

	}
	
	return sz_rez;
}
//-----------------------------------------------------------------
//INPUT:
// d_mtrxSig - input matrix, dimentions nchan x (lenChunk * npol) 
// lenChunk - number comlex elements in time series
// nchan - number of channels
// npol - number of polarizations, =2 || 4
// OUTPUT:
// pcmparrRawSignalCur - matrix with raw signal, corresponding with chunk

__global__
void unpackInput(cufftComplex* pcmparrRawSignalCur, inp_type_* d_parrInput, const int  lenChunk
	, const int  nchan, const int  npol)
{
	const int inChan = blockIdx.y;

	const int inBlockCur = blockIdx.x;

	unsigned int  numElemColOut = inBlockCur * blockDim.x + threadIdx.x;
	unsigned int  numElemColInp = npol * numElemColOut;
	if (numElemColOut >= lenChunk)
	{
		return;
	}
	for (int i = 0; i < npol / 2; ++i)
	{

		pcmparrRawSignalCur[(inChan * npol / 2 + i) * lenChunk + numElemColOut].x
			= (float)d_parrInput[inChan * lenChunk * npol + numElemColInp + 2 * i];
		pcmparrRawSignalCur[(inChan * npol / 2 + i) * lenChunk + numElemColOut].y
			= (float)d_parrInput[inChan * lenChunk * npol + numElemColInp + 2 * i + 1];
	}

}
//---------------------------------------------------
bool CBlock::fncChunkProcessing_gpu(cufftComplex* pcmparrRawSignalCur
	, void* pAuxBuff_fdmt	
	, cufftComplex* pcarrTemp
	, cufftComplex* pcarrCD_Out
	, cufftComplex* pcarrBuff
	, float* pAuxBuff_flt, fdmt_type_* d_arrfdmt_norm
	, const int IDeltaT, cufftHandle plan0, cufftHandle plan1
	, structOutDetection* pstructOut
    , float *pcoherentDedisp)
{
	// 1. installation of pointers	for pAuxBuff_the_rest	

	fdmt_type_* d_parr_fdmt_inp = (fdmt_type_ * )pAuxBuff_flt; //4	
	fdmt_type_* d_parr_fdmt_out = (fdmt_type_*)pAuxBuff_flt + m_lenChunk; //5

	
	
	//// !1

	// std::vector<std::complex<float>> data(LEnChunk, 0);
	//cudaMemcpy(data.data(), pffted_rowsignal, LEnChunk * sizeof(std::complex<float>),
	//	cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	//std::array<long unsigned, 1> leshape127{ LEnChunk };
	//npy::SaveArrayAsNumpy("ffted.npy", false, leshape127.size(), leshape127.data(), data);
	auto start = std::chrono::high_resolution_clock::now();

	cufftResult result;
	// 2. create FFT	
	checkCudaErrors(cufftExecC2C(plan0, pcmparrRawSignalCur, pcmparrRawSignalCur, CUFFT_FORWARD));
	
	// !2

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	iFFT_time += duration.count();

	////std::vector<std::complex<float>> data(LEnChunk, 0);
	////cudaMemcpy(data.data(), pffted_rowsignal, LEnChunk * sizeof(std::complex<float>),
	////	cudaMemcpyDeviceToHost);
	////cudaDeviceSynchronize();
	

	// 3.		
	float valConversionConst = DISPERSION_CONSTANT * (1. / (m_Fmin * m_Fmin) - 1. / (m_Fmax * m_Fmax)) * (m_Fmax - m_Fmin);
	float valN_d = m_d_max * valConversionConst;
	const int N_p = m_len_sft * m_nchan;
	int n_coherent = int(ceil(valN_d / (N_p * N_p)));
	cout << " n_coherent = " << n_coherent << endl;
	// !3

	
	structOutDetection* pstructOutCur = NULL;
	checkCudaErrors(cudaMallocManaged((void**)&pstructOutCur, sizeof(structOutDetection)));
	cudaDeviceSynchronize();
	pstructOutCur->snr = 1. - FLT_MAX;
	pstructOutCur->icol = -1;
	pstructOut->snr = m_sigma_bound;
	//// 4. main loop
	const int IMaxDT = N_p;
	
	float coherent_d = -1.;
	bool breturn = false;
	for (int iouter_d = 0; iouter_d < n_coherent; ++iouter_d)
	{
		cout << "coherent iteration " << iouter_d << endl;

		double valcur_coherent_d = (( double)iouter_d) * (( double)m_d_max / (( double)n_coherent));
		cout << "cur_coherent_d = " << valcur_coherent_d << endl;

			/*std::vector<std::complex<float>> data(m_lenChunk * m_nchan * m_npol /2, 0);
			cudaMemcpy(data.data(), pcmparrRawSignalCur, m_lenChunk * m_nchan * m_npol / 2 * sizeof(std::complex<float>),
			cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();*/
			
		// fdmt input matrix computation
		calcFDMT_Out_gpu(d_parr_fdmt_out, pcmparrRawSignalCur, pcarrCD_Out
			, pcarrTemp, d_parr_fdmt_inp
			, IMaxDT, DISPERSION_CONSTANT * valcur_coherent_d
			, pAuxBuff_fdmt, IDeltaT, plan0, plan1, pcarrBuff);		
		// !
		/*float* parr_fdmt_out = (float*)malloc(m_lenChunk * sizeof(float));
		cudaMemcpy(parr_fdmt_out, d_parr_fdmt_out, m_lenChunk * sizeof(float), cudaMemcpyDeviceToHost);
		float valmax = -0., valmin = 0.;
		unsigned int iargmax = -1, iargmin = -1;
		findMaxMinOfArray(parr_fdmt_out, m_lenChunk, &valmax, &valmin
			, &iargmax, &iargmin);


		float* arrfdmt_norm = (float*)malloc(m_lenChunk * sizeof(float));
		cudaMemcpy(arrfdmt_norm, d_arrfdmt_norm, m_lenChunk * sizeof(float), cudaMemcpyDeviceToHost);
		float valmax1 = -0., valmin1 = 0.;
		unsigned int iargmax1 = -1, iargmin1 = -1;
		findMaxMinOfArray(arrfdmt_norm, m_lenChunk, &valmax1, &valmin1
			, &iargmax1, &iargmin1);


		free(parr_fdmt_out);
		free(arrfdmt_norm);*/

		
		const int Rows = m_len_sft * m_nchan;
		const int Cols = m_lenChunk / Rows;
		const dim3 blockSize(1024, 1, 1);
		const dim3 gridSize((Cols + blockSize.x - 1) / blockSize.x, Rows, 1);
		float* d_pAuxArray = (float*)d_parr_fdmt_inp;
		int* d_pAuxNumArray = (int*)(d_pAuxArray + gridSize.x * gridSize.y);
		int* d_pWidthArray = d_pAuxNumArray + +gridSize.x * gridSize.y;
		detect_signal_gpu(d_parr_fdmt_out, d_arrfdmt_norm, Rows
			, Cols, m_length_sum_wnd, gridSize, blockSize
			, d_pAuxArray, d_pAuxNumArray, d_pWidthArray, pstructOutCur);
		if ((*pstructOutCur).snr >= (*pstructOut).snr)
		{
			(*pstructOut).snr = (*pstructOutCur).snr;
			(*pstructOut).icol = (*pstructOutCur).icol;
			(*pstructOut).irow = (*pstructOutCur).irow;
			(*pstructOut).iwidth = (*pstructOutCur).iwidth;
			
			*pcoherentDedisp = valcur_coherent_d;			

			std::cout << "SNR = " << (*pstructOut).snr << endl;
			std::cout << "ROW ARGMAX = " << (*pstructOut).irow << endl;
			std::cout << "COLUMN ARGMAX = " << (*pstructOut).icol << endl;

			int frequency = 1500; // Frequency in hertz
			int duration = 500;   // Duration in milliseconds
			emitSound(frequency, duration);
			emitSound(frequency+500, duration);
			d_pAuxArray = NULL;
			d_pAuxNumArray = NULL;
			d_pWidthArray = NULL;
			

			breturn = true;
		}
		
		/*std::vector<float> data(LEnChunk, 0);
		cudaMemcpy(data.data(), parr_fdmt_out, LEnChunk * sizeof(float),
			cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();*/
		
		
		
	}
	cudaFree(pstructOutCur);
	return breturn;
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
int CBlock::calcFDMT_Out_gpu(fdmt_type_* parr_fdmt_out, cufftComplex* pffted_rowsignal, cufftComplex* pcarrCD_Out
	, cufftComplex* pcarrTemp,  fdmt_type_* d_parr_fdmt_inp
	, const unsigned int IMaxDT, const  double VAl_practicalD
	, void* pAuxBuff_fdmt, const int IDeltaT, cufftHandle plan0, cufftHandle plan1, cufftComplex* pAuxBuff)
{
	/*std::vector<std::complex<float>> data(m_lenChunk * m_nchan * m_npol / 2, 0);
	cudaMemcpy(data.data(), pffted_rowsignal, m_lenChunk * m_nchan * m_npol / 2 * sizeof(std::complex<float>),
		cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();*/

	fncCD_Multiband_gpu(pcarrCD_Out, pffted_rowsignal
		, VAl_practicalD, plan0, pcarrTemp);

	auto start = std::chrono::high_resolution_clock::now();

	cufftResult result = cufftExecC2C(plan1, pcarrCD_Out, pcarrTemp, CUFFT_FORWARD);
	cudaDeviceSynchronize();

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	iFFT_time += duration.count();
	//

	calc_fdmt_inp(d_parr_fdmt_inp, pcarrTemp, (float*)pcarrCD_Out);	
	//

	start = std::chrono::high_resolution_clock::now();
	fncFdmtU_cu(d_parr_fdmt_inp      // on-device input image
		, pAuxBuff_fdmt
		, m_len_sft * m_nchan
		, m_lenChunk / (m_len_sft * m_nchan) // dimensions of input image 	
		, IDeltaT
		, m_Fmin
		, m_Fmax
		, IMaxDT
		, parr_fdmt_out	// OUTPUT image, dim = IDeltaT x IImgcols
		, false
	);
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	iFdmt_time += duration.count();
	return 0;
}

//--------------------------------------------------------------------
//INPUT:
//1. pcarrTemp - complex array with total length  = m_lenChunk * (m_npol/2)* m_nchan
// pcarrTemp can be interpreted as matrix, consisting of  m_nchan *(m_npol/2) rows
// each row consists of m_len_sft subrows corresponding to m_len_sft subfrequencies
// 2.pAuxBuff - auxillary buffer to compute mean and dispersions of each row ofoutput matrix d_parr_fdmt_inp
//OUTPUT:
//d_parr_fdmt_inp - matrix with dimensions (m_nchan*m_len_sft) x (m_lenChunk/m_len_sft)
// d_parr_fdmt_inp[i][j] = 
//
void CBlock::calc_fdmt_inp(fdmt_type_* d_parr_fdmt_inp, cufftComplex* pcarrTemp
	, float*pAuxBuff)
{	
	int nRows = m_len_sft * m_nchan;
	int nCols = m_lenChunk / nRows;
	dim3 threadsPerBlock(TILE_DIM, TILE_DIM, 1);
	dim3 blocksPerGrid((m_len_sft + TILE_DIM - 1) / TILE_DIM, (m_lenChunk / m_len_sft + TILE_DIM - 1) / TILE_DIM, m_nchan);
	size_t sz = TILE_DIM * (TILE_DIM + 1) * sizeof(float);
	float* d_parr_fdmt_inp_flt = pAuxBuff;	
	calcPowerMtrx_kernel << < blocksPerGrid, threadsPerBlock, sz >> > (d_parr_fdmt_inp_flt, m_lenChunk/ m_len_sft, m_len_sft, m_npol, pcarrTemp);
	cudaDeviceSynchronize();
	/*std::vector<float> data0(m_lenChunk, 0);
	cudaMemcpy(data0.data(), d_parr_fdmt_inp_flt, m_lenChunk * sizeof(float),
		cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();*/
	
	float* d_arrRowMean = (float*)pcarrTemp;
	float* d_arrRowDisp = d_arrRowMean + nRows;
	
	
	auto start = std::chrono::high_resolution_clock::now();

	// Calculate mean and variance
	float* pval_mean = d_arrRowDisp + nRows;
	float* pval_stdDev = pval_mean + 1;
	float* pval_dispMean = pval_stdDev + 1;
	float* pval_dispStd = pval_dispMean + 1;
	

	blocksPerGrid = nRows;
	int treadsPerBlock = calcThreadsForMean_and_Disp(nCols);
	sz = (2 * sizeof(float) + sizeof(int)) * treadsPerBlock;
	// 1. calculations mean values and dispersions for each row of matrix d_parr_fdmt_inp_flt
	// d_arrRowMean - array contents  mean values of each row of input matrix pcarrTemp
	// d_arrRowDisp - array contents  dispersions of each row of input matrix pcarrTemp
	
	calcRowMeanAndDisp << < blocksPerGrid, treadsPerBlock, sz >> > (d_parr_fdmt_inp_flt, nRows, nCols, d_arrRowMean, d_arrRowDisp);
	cudaDeviceSynchronize();

	//std::vector<float> data0(nRows, 0);
	//cudaMemcpy(data0.data(), d_arrRowDisp, nRows * sizeof(float),
	//	cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();

	//std::vector<float> data1(nRows, 0);
	//cudaMemcpy(data1.data(), d_arrRowMean, nRows * sizeof(float),
	//	cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();

	//float* parr_fdmt_inp_flt = (float*)malloc(nRows* nCols *sizeof(float));
	//cudaMemcpy(parr_fdmt_inp_flt, d_parr_fdmt_inp_flt, nRows * nCols * sizeof(float), cudaMemcpyDeviceToHost);
	//float* arrM = (float*)malloc(nRows * sizeof(float));
	//float* arrD = (float*)malloc(nRows * sizeof(float));
	//memset(arrM, 0, nRows * sizeof(float));
	//memset(arrD, 0, nRows * sizeof(float));
	//for (int i = 0; i < nRows; ++i)
	//{
	//	for (int j = 0; j < nCols; ++j)
	//	{
	//		arrM[i] += parr_fdmt_inp_flt[i * nCols + j];
	//		arrD[i] += parr_fdmt_inp_flt[i * nCols + j] * parr_fdmt_inp_flt[i * nCols + j];
	//	}
	//	arrM[i] = arrM[i] / ((float)nCols);
	//	arrD[i] = arrD[i] / ((float)nCols) - arrM[i] * arrM[i];

	//}


	//free(parr_fdmt_inp_flt);
	//free(arrM);
	//free(arrD);
	// 2. calculations mean value and standart deviation for full matrix pcarrTemp
	// it is demanded to normalize matrix pcarrTemp
	blocksPerGrid = 1;
	treadsPerBlock = calcThreadsForMean_and_Disp(nRows);
	sz = treadsPerBlock * (2 * sizeof(float) + sizeof(int));
	kernel_OneSM_Mean_and_Std << <blocksPerGrid, treadsPerBlock, sz >> > (d_arrRowMean, d_arrRowDisp, nRows
		, pval_mean, pval_stdDev);
	cudaDeviceSynchronize();

	//

	//float mean = -1., disp = -1.;
	//cudaMemcpy(&mean, pval_mean, sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&disp, pval_stdDev, sizeof(float), cudaMemcpyDeviceToHost);

	//// check up
	//float* arrmean = (float*)malloc(nRows * sizeof(float));
	//float* arrdisp = (float*)malloc(nRows * sizeof(float));
	//cudaMemcpy(arrmean, d_arrRowMean, nRows * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(arrdisp, d_arrRowDisp, nRows * sizeof(float), cudaMemcpyDeviceToHost);
	//float sum = 0.;
	//for (int i = 0; i < nRows; ++i)
	//{
	//	sum += arrmean[i];
	//}
	//sum = sum / ((float)nRows);

	//float disp1 = 0;
	//for (int i = 0; i < nRows; ++i)
	//{
	//	disp1 += arrmean[i] * arrmean[i] + arrdisp[i];// (arrmean[i] - sum)* (arrmean[i] - sum);
	//}
	//disp1 = disp1/ ((float)nRows) - sum*sum;

	//free(arrmean);
	//free(arrdisp);


	// 3. calculations mean value and standart deviation for array d_arrRowDisp
	// it is demanded to clean out tresh from matrix pcarrTemp
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	iMeanDisp_time += duration.count();
	
	
	
	int threads = 128;
	calculateMeanAndSTD_for_oneDimArray_kernel << <1, threads, threads * 2 * sizeof(float) >> > (d_arrRowDisp, nRows, pval_dispMean, pval_dispStd);
	cudaDeviceSynchronize();

	//float hval_dispMean = -1;
	//float hval_dispStd = -1;
	//cudaMemcpy(&hval_dispMean, pval_dispMean, sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&hval_dispStd, pval_dispStd, sizeof(float), cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	// 4.Clean and normalize array
	const dim3 blockSize(256, 1, 1);
	const dim3 gridSize(1, nRows, 1);
	
	

	normalize_and_clean << < gridSize, blockSize >> >
		(d_parr_fdmt_inp, d_parr_fdmt_inp_flt, nRows, nCols
		, pval_mean, pval_stdDev, d_arrRowDisp, pval_dispMean, pval_dispStd  );	
	cudaDeviceSynchronize();

	//float* parr_fdmt_inp = (float*)malloc(nRows * nCols * sizeof(float));
	//cudaMemcpy(parr_fdmt_inp, d_parr_fdmt_inp, nRows * nCols * sizeof(float), cudaMemcpyDeviceToHost);

	//float valmax = -0., valmin = 0.;
	//unsigned int iargmax = -1, iargmin = -1;
	//findMaxMinOfArray(parr_fdmt_inp, nRows * nCols, &valmax,  &valmin
	//	, &iargmax, &iargmin);

	//auto end1 = std::chrono::high_resolution_clock::now();
	//auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - end);
	//iNormalize_time += duration1.count();
	//free(parr_fdmt_inp);

	d_arrRowMean = NULL;
	d_arrRowDisp = NULL;
	pval_mean = NULL;	
	pval_stdDev = NULL;
	pval_dispMean = NULL;
	pval_dispStd = NULL;

	

}

//-------------------------------------------------------------------
__device__
float fnc_norm2(cufftComplex* pc)
{
	return ((*pc).x * (*pc).x + (*pc).y * (*pc).y);
}

//-------------------------------------------------------------------
void CBlock::fncCD_Multiband_gpu(cufftComplex* pcarrCD_Out, cufftComplex* pcarrffted_rowsignal
	, const  double VAl_practicalD, cufftHandle  plan, cufftComplex* pAuxBuff)
{
	 double step = (( double)m_Fmax - ( double)m_Fmin) / (( double)m_lenChunk)/((double)m_nchan);
	
	const dim3 blockSize = dim3(1024, 1, 1);
	const dim3 gridSize = dim3((m_lenChunk + blockSize.x - 1) / blockSize.x, m_npol/2, m_nchan);	
	kernel_ElementWiseMult << < gridSize, blockSize/*, sz*/ >> >
		(pAuxBuff,pcarrffted_rowsignal, m_lenChunk, step, VAl_practicalD, m_Fmin
			, m_Fmax);	
	cudaDeviceSynchronize();

	/*std::vector<std::complex<float>> data(m_lenChunk * m_nchan * m_npol / 2, 0);
	cudaMemcpy(data.data(), pAuxBuff, m_lenChunk * m_nchan * m_npol / 2 * sizeof(std::complex<float>),
		cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();*/

	auto start = std::chrono::high_resolution_clock::now();
	checkCudaErrors(cufftExecC2C(plan, pAuxBuff, pcarrCD_Out, CUFFT_INVERSE));

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	iFFT_time += duration.count();

	/*std::vector<std::complex<float>> data1(m_lenChunk * m_nchan * m_npol / 2, 0);
	cudaMemcpy(data1.data(), pcarrCD_Out, m_lenChunk * m_nchan * m_npol / 2 * sizeof(std::complex<float>),
		cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();*/
	
	scaling_kernel << <1, 1024, 0 >> > (pcarrCD_Out, m_lenChunk * m_npol* m_nchan/2, 1.f / ((float)m_lenChunk));
	cudaDeviceSynchronize();
}
//-------------------------------------------------------------
 // calculation:
 // H = np.e**(-(2*np.pi*complex(0,1) * practicalD /(f_min + f) + 2*np.pi*complex(0,1) * practicalD*f /(f_max**2)))
 // np.fft.fft(raw_signal) * H 
// gridDim.z - num channels
// gridDim.y - num polarizations in physical sense
__global__ void kernel_ElementWiseMult(cufftComplex* pAuxBuff, cufftComplex* pcarrffted_rowsignal
	, const unsigned int LEnChunk, const  double step, const  double VAl_practicalD, const double Fmin
	, const double Fmax)
{
	__shared__ double arrf[2];
	__shared__ int iarr[1];

	
	double chanBW = (Fmax - Fmin) / gridDim.z;
	arrf[0] = Fmin + chanBW * blockIdx.y;
	arrf[1] = arrf[0] + chanBW;
	iarr[0] = (blockIdx.z * gridDim.y + blockIdx.y) * LEnChunk;

	
	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= LEnChunk)
	{
		return;
	}
	double t = VAl_practicalD * (1. / (arrf[0] + step * (double)j) + 1. / (arrf[1]) * (step * (double)j / arrf[1]));

	double val_prD_int = 0, val_prD_frac = 0;
	double t4 = -modf(t, &val_prD_int) * 2.0;

	double val_x = 0.;
	double val_y = 0.;
	sincospi(t4, &val_y, &val_x);
	unsigned int nelem = iarr[0] + j;	
	pAuxBuff[nelem].x = (float)(val_x * pcarrffted_rowsignal[nelem].x - val_y * pcarrffted_rowsignal[nelem].y); // Real part
	pAuxBuff[nelem].y =  (float)(val_x * pcarrffted_rowsignal[nelem].y + val_y * pcarrffted_rowsignal[nelem].x); // Imaginary part

}

//---------------------------------------------------------------
__global__
void scaling_kernel(cufftComplex* data, long long element_count, float scale)
{
	const int tid = threadIdx.x;
	const int stride = blockDim.x;
	for (long long i = tid; i < element_count; i += stride)
	{
		data[i].x *= scale;
		data[i].y *= scale;
	}
}
//----------------------------------------------------

__global__
void calcMultiTransposition_kernel(fdmt_type_* output, const int height, const int width, fdmt_type_* input)
{
	__shared__ fdmt_type_ tile[TILE_DIM][TILE_DIM + 1]; // Shared memory tile
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int ichan = blockIdx.z;
	// Transpose data from global to shared memory
	if (x < width && y < height)
	{
		tile[threadIdx.y][threadIdx.x] = input[ichan * height * width + y * width + x];
	}
	__syncthreads();

	// Calculate new indices for writing to output
	x = blockIdx.y * TILE_DIM + threadIdx.x;
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	// Transpose data from shared to global memory
	if (x < height && y < width)
	{
		output[ichan * height * width + y * height + x] = tile[threadIdx.x][threadIdx.y];
	}
}
//------------------------------------------

__global__
void calcPartSum_kernel(fdmt_type_* d_parr_out, const int lenChunk, const int npol_physical, cufftComplex* d_parr_inp)
{

	int ichan = blockIdx.y;
	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind < lenChunk)
	{
		fdmt_type_ sum = 0;
		for (int i = 0; i < npol_physical; ++i)
		{
			sum += fnc_norm2(&d_parr_inp[(ichan * npol_physical + i) * lenChunk + ind]);
		}
		d_parr_out[ichan * lenChunk + ind] = sum;
	}
}
//------------------------------------------
__global__
void calcPowerMtrx_kernel(float* output, const int height, const int width, const int npol, cufftComplex* input) 
{
	__shared__ float tile[TILE_DIM][TILE_DIM + 1]; // Shared memory tile

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int ichan = blockIdx.z;
	// Transpose data from global to shared memory
	if (x < width && y < height)
	{
		float sum = 0.;
		for (int i = 0; i < npol; ++i)
		{
			sum += fnc_norm2(&input[(ichan * npol + i) * height * width + y * width + x]);
		}

		tile[threadIdx.y][threadIdx.x] = sum;
	}
	__syncthreads();

	// Calculate new indices for writing to output
	x = blockIdx.y * TILE_DIM + threadIdx.x;
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	// Transpose data from shared to global memory
	if (x < height && y < width) {
		output[ichan * height * width + y * height + x] = tile[threadIdx.x][threadIdx.y];
	}
}
//---------------------------------------------------------------

__global__ 
void normalize_and_clean(fdmt_type_* parrOut, float* d_arr, const int NRows, const int NCols
	,float *pmean, float *pstd, float* d_arrRowDisp, float *pmeanDisp, float *pstdDisp)
{
	__shared__ int sbad[1];
	unsigned int i = threadIdx.x;
	unsigned int irow = blockIdx.y;
	if (i >= NCols)
	{
		return;
	}
	if (fabs(d_arrRowDisp[irow] - *pmeanDisp) > 4. * (*pstdDisp))
	{
		sbad[0] = 1;
	}
	else
	{
		sbad[0] = 0;
	}
	//--------------------------------
	if (sbad[0] == 1)
	{
		while (i < NCols)
		{
			parrOut[irow * NCols + i] = 0;
			i += blockDim.x;
		}
	}
	else
	{
		while (i < NCols)
		{
			parrOut[irow * NCols + i] = (fdmt_type_)((d_arr[irow * NCols + i] - (*pmean) )/((*pstd )));
			i += blockDim.x;
		}
	}
	

}

//--------------------------------------------------------------
//
//__global__ void kernel_clean_and_normalize_input_fdmt_array(fdmt_type_* parrOut, const unsigned int nCols
//	,const float d_mean,const float* d_std, float* parrInp, float* d_arrRowDisp,const float d_val_dispMean,const float* d_val_dispStd)
//{
//	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
//	unsigned int i = blockIdx.y;
//	__shared__ int b[1];
//	if (fabsf(d_arrRowDisp[i] - d_val_dispMean) > 4. * d_val_dispStd)
//	{
//		b[0] = 0;
//	}
//	else
//	{
//		b[0] = true;
//	}
//	__syncthreads();
//
//	if (j >= len)
//	{
//		return;
//	}
//	if (b[0])
//	{
//		parrOut[j] = (fdmt_type_)((parrInp[i] - (*pmean)) / ((*pDisp) * 0.25));
//	}
//	else
//	{
//		parrOut[j] = 0;
//	}
//
//}
//-----------------------------------------------------------------
//__global__
//void fncSignalDetection_gpu(fdmt_type_* parr_fdmt_out, fdmt_type_* parrImNormalize, const unsigned int qCols
//	, const unsigned int len, fdmt_type_* pmaxElement, unsigned int* argmaxRow, unsigned int* argmaxCol)
//{
//	extern __shared__ char sbuff1[];
//	fdmt_type_* sdata = (fdmt_type_*)sbuff1;
//	int* sNums = (int*)((char*)sbuff1 + blockDim.x * sizeof(fdmt_type_));
//
//	unsigned int tid = threadIdx.x;
//	unsigned int i = tid;
//	if (tid >= len)
//	{
//		sNums[tid] = -1;
//		sdata[tid] = -FLT_MAX;
//	}
//	else
//	{
//		fdmt_type_ localMax = -FLT_MAX;
//		int localArgMax = 0;
//
//		// Calculate partial sums within each block   
//		while (i < len)
//		{
//			fdmt_type_ t = parr_fdmt_out[i] / sqrt(parrImNormalize[i] * 16 + 0.000001);
//			parr_fdmt_out[i] = t;
//			if (t > localMax)
//			{
//				localMax = t;
//				localArgMax = i;
//			}
//			i += blockDim.x;
//		}
//		// Store partial sums in shared memory
//		//numLocal = len / blockDim.x;
//		sNums[tid] = localArgMax;
//		sdata[tid] = localMax;
//	}
//	__syncthreads();
//
//	// Parallel reduction within the block to sum partial sums
//	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
//	{
//		if (tid < s)
//		{
//			if (sdata[tid + s] > sdata[tid])
//			{
//				sdata[tid] = sdata[tid + s];
//				sNums[tid] = sNums[tid + s];
//			}
//
//		}
//		__syncthreads();
//	}
//	// Only thread 0 within each block computes the block's sum
//	if (tid == 0)
//	{
//		*pmaxElement = sdata[0];
//		*argmaxRow = sNums[0] / qCols;
//		*argmaxCol = sNums[0] % qCols;
//
//	}
//	__syncthreads();
//}






