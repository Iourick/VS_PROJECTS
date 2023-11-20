
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "FdmtCu0.cuh"

#include <math.h>
#include <stdio.h>
#include <array>
#include <iostream>
#include <string>

#include <vector>
#include <cstdlib> // For random value generation
#include <ctime>   // For seeding the random number generator
#include "npy.hpp"
#include <algorithm> 
#include "kernel.cuh"
#include <chrono>
#include "fileInput.h"
#include "DrawImg.h"
#include "Constants.h"

#include "StreamParams.h"
#include "utilites.h"
#include "read_and_write_log.h"
#include "HybridDedispersionStream_gpu.cuh"








using namespace std;

class StreamParams;
const char chStrDefaultInputPass[] = "..//HYBRID_TESTS//data.bin";
//const char chStrDefaultInputPass[] = "..//HYBRID_TESTS//data3.bin";
int numAttemptions = 0;

void printDeviceInfo()
{
	int deviceId;
	int numberOfSMs;
	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
	cudaDeviceProp deviceProps;
	cudaGetDeviceProperties(&deviceProps, deviceId);
	std::string deviceName = deviceProps.name;
	std::cout << "Device Name: " << deviceName << std::endl;
	std::cout << "Number of SM: " << numberOfSMs << std::endl;
}

int dialIntroducing()
{
	// 1. define path to data file with complex time serie
	std::cout << "By default input file is  " << chStrDefaultInputPass << endl;//  \"D://MyVSprojPy//hybrid//data.bin\"" << endl;
	std::cout << "if you want default, print y, otherwise n" << endl;
	char userInput[200] = { 0 };
	char ch = std::cin.get();
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');


	if (ch == 'y')
	{
		strcpy(mchInpFilePass, chStrDefaultInputPass);
	}
	else
	{
		std::cout << "Enter the pass:" << endl;//  with double quotation marks \"..\"" << endl;
		std::cin.getline(userInput, 200);
		strcpy(mchInpFilePass, userInput);
	}
	// 1!


	// 2. reading header of input file


	if (readHeader(mchInpFilePass, mlenarr, m_n_p
		, mvalD_max, mvalf_min, mvalf_max, mvalSigmaBound) == 1)
	{
		std::cerr << "Error opening file." << std::endl;
		return 1;
	}
	// 2 !

	// 3. printing header's info
	std::cout << "Header's information:" << endl;
	std::cout << "Length of time serie = " << mlenarr << endl;
	// 3!

	// 4. default parametres

	mnumBegin = 1;
	mnumEnd = mlenarr;
	mlenChunk = pow(2, 20);
	std::cout << "By default parametres:" << endl;
	std::cout << "Length of chunk( 2 **20 ) = " << mlenChunk << endl;
	std::cout << "Number of first elem =  " << mnumBegin << endl;
	std::cout << "Number of last elem =  " << mnumEnd << endl;
	std::cout << "If you want go on by default print y, otherwise print n " << endl;

	char ch1 = std::cin.get();
	if (ch1 != 'y')
	{
		for (int i = 0; i < 4; ++i)
		{
			std::cout << "Print begin number of time serie: ";
			std::cin >> mnumBegin;

			std::cout << "Print end number of time serie: ";
			std::cin >> mnumEnd;

			std::cout << "Print chunk's length: ";
			std::cin >> mlenChunk;

			if ((mnumBegin < 1) || (mnumEnd > mlenarr) || (mlenChunk > (mnumEnd - mnumBegin)))
			{
				std::cout << "Check up parametres" << endl;
				++numAttemptions;
				if (numAttemptions == 4)
				{
					return 2;
				}
			}
			else
			{
				break;
			}
		}
	}
	// 4!
	mnumBegin -= 1;
	mnumEnd -= 1;
	return 0;
}

//---------------------------------------

int main(int argc, char** argv)
{
	constexpr int size = 5; // Define the size of your arrays

	float arrA[size] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
	float arrB[size] = { 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
	float result[size];

	float* d_a, * d_b, * d_c;

	cudaMalloc((void**)&d_a, size * sizeof(float));
	cudaMalloc((void**)&d_b, size * sizeof(float));
	cudaMalloc((void**)&d_c, size * sizeof(float));

	cudaMemcpy(d_a, arrA, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, arrB, size * sizeof(float), cudaMemcpyHostToDevice);

	constexpr int blockSize = 256;
	int gridSize = (size + blockSize - 1) / blockSize;

	vectorAdd << <gridSize, blockSize >> > (d_a, d_b, d_c, size);

	cudaMemcpy(result, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	std::cout << "Resultant array after addition: ";
	for (int i = 0; i < size; ++i) {
		std::cout << result[i] << " ";
	}
	std::cout << std::endl;

	return 0;




	dialIntroducing();
	// 5. Create member of class CStreamParams
	CStreamParams* pStreamPars = new CStreamParams(mchInpFilePass, mnumBegin, mnumEnd, mlenChunk);
	// 5!

	// 6. memory allocation for output information 
	int* piarrNumSucessfulChunks = (int*)malloc(sizeof(int) * (1 + (mnumEnd - mnumBegin) / mlenChunk));
	float* parrCoherent_d = (float*)malloc(sizeof(float) * (1 + (mnumEnd - mnumBegin) / mlenChunk));
	int quantOfSuccessfulChunks = 0;
	// 6 !

	// 7. call function processing the stream    
	//int irez = fncHybridScan(nullptr, piarrNumSucessfulChunks, parrCoherent_d, quantOfSuccessfulChunks, pStreamPars);
	int irez = fncHybridDedispersionStream(piarrNumSucessfulChunks, parrCoherent_d, quantOfSuccessfulChunks, pStreamPars);

	fncWriteLog_("info.log", mchInpFilePass, "hybrid dedispersion, C++ implementation"
		, mlenChunk, quantOfSuccessfulChunks, piarrNumSucessfulChunks, parrCoherent_d, 0);

	// 7!

	// 8. report
	std::cout << "------------ Calculations completed successfully -------------" << endl;
	std::cout << "Pass to Data File : " << mchInpFilePass << endl;
	std::cout << "Successful Chunks Number : " << quantOfSuccessfulChunks << endl;
	std::cout << "Chunk Num., Coh. Disp. : " << endl;
	for (int i = 0; i < quantOfSuccessfulChunks; ++i)
	{
		std::cout << piarrNumSucessfulChunks[i] << " ; " << parrCoherent_d[i] << endl;
	}

	free(piarrNumSucessfulChunks);
	free(parrCoherent_d);
	delete pStreamPars;

	std::cout << "Running Time = " << 0. << "ms" << endl;
	std::cout << "---------------------------------------------------------" << endl;

	char chInp[200] = { 0 };
	std::cout << "if you  want to quit, print q" << endl;
	std::cout << "if you want to proceed, print y " << endl;

	std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	char ch0 = std::cin.get();
	if (ch0 == 'q')
	{
		return 0;
	}
	//const int dataSize = 8;  // Size of the data array
	//const int dataSizeComplex = dataSize / 2 + 1; // Size of the complex output

	//// Initialize input data on the host (CPU)
	//float inputData[dataSize] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f };

	//// Allocate device memory for input and output data
	//float* d_inputData;
	//cufftComplex* d_outputData;

	//cudaMalloc((void**)&d_inputData, dataSize * sizeof(float));
	//cudaMalloc((void**)&d_outputData, dataSizeComplex * sizeof(cufftComplex));

	//// Copy input data from host to device
	//cudaMemcpy(d_inputData, inputData, dataSize * sizeof(float), cudaMemcpyHostToDevice);

	//// Create cuFFT plan
	//cufftHandle plan;
	//cufftPlan1d(&plan, dataSize, CUFFT_R2C, 1);  // Real to complex FFT

	//// Execute FFT
	//cufftExecR2C(plan, d_inputData, d_outputData);

	//// Copy the result back from device to host
	//cufftComplex* outputData = new cufftComplex[dataSizeComplex];
	//cudaMemcpy(outputData, d_outputData, dataSizeComplex * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	//// Display the FFT results
	//std::cout << "FFT result:" << std::endl;
	//for (int i = 0; i < dataSizeComplex; ++i) {
	//	std::cout << "Element " << i << ": " << outputData[i].x << " + " << outputData[i].y << "i" << std::endl;
	//}

	//// Clean up
	//cufftDestroy(plan);
	//cudaFree(d_inputData);
	//cudaFree(d_outputData);
	//delete[] outputData;

	//return 0;
	
	printDeviceInfo();


//--------------------------------------------------------------------------------------------------------------
//------------------- prepare to work -------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
// initiate pointer to input image
	//int* piarr = (int*)malloc(sizeof(int));

	//// initiate 2-pointer to input image, in order to realloc memory to satisfy arbitrary dimensions
	//int** ppiarrImage = &piarr;

	//// initiating input variables
	//int iMaxDT = 0;
	//int iImRows = 0, iImCols = 0;
	//float val_fmax = 0., val_fmin = 0.;

	//// reading input files from folder 
	//int ireturn = downloadInputData(strInpFolder, &iMaxDT, ppiarrImage, &iImRows, &iImCols,
	//	&val_fmin, &val_fmax);

	//// analysis output of reading function
	//switch (ireturn)
	//{
	//case 1:
	//	cout << "Err. ...Can't allocate memory for input image. Oooops... " << std::endl;
	//	return 1;
	//case 2:
	//	cout << "Err. ...Input dimensions must be a power of 2. Oooops... " << std::endl;
	//	return 1;
	//case 0:
	//	cout << "Input data downloaded properly " << std::endl;
	//	break;
	//default:
	//	cout << "..something extraordinary happened! Oooops..." << std::endl;
	//	break;
	//}

	//// 5.1
	//if ((iImRows == 1024) && (BDIM_512_1024))
	//{
	//	iImRows = 512;
	//	iMaxDT = 512;
	//	piarr = (int*)realloc(piarr, iImRows * iImCols * sizeof(int));

	//}
	//// ! 5.1

	//// declare constants
	//const int IMaxDT = iMaxDT;
	//const int IImgrows = iImRows;
	//const int IImgcols = iImCols;
	//const float VAlFmin = val_fmin;
	//const float VAlFmax = val_fmax;


	//// handy pointer to input image
	//int* piarrImage = *ppiarrImage;

	//

	////--------------------------------------------------------------------------------------------------------------
	////-------------------- end of prepare ------------------------------------------------------------------------------------------
	////------------------- begin to calculate cuda var -------------------------------------------------------------------------------------------
	////--------------------------------------------------------------------------------------------------------------
	//// 1. allocate memory for device array
	//

	//int* u_piarrImOut = 0;
	//cudaMallocManaged(&u_piarrImOut, IImgcols * IMaxDT * sizeof(int));

	//int* d_piarrImage = 0;
	////cudaMallocManaged(&d_piarrImage, IImgcols * IImgrows * sizeof(int));
	//cudaMalloc(&d_piarrImage, IImgcols * IImgrows * sizeof(int));

	//
	//// !1


	//// 1. quant iteration's calculation
	//const int I_F = (int)(log2((double)(IImgrows)));
	//// !1

	//// 2. temp variables calculations
	//const float VAl_dF = (VAlFmax - VAlFmin) / ((float)(IImgrows));
	//const int IDeltaT = int(ceil((IMaxDT - 1.) * (1. / (VAlFmin * VAlFmin)
	//	- 1. / ((VAlFmin + VAl_dF) * (VAlFmin + VAl_dF)))
	//	/ (1. / (VAlFmin
	//		* VAlFmin) - 1. / (VAlFmax * VAlFmax))));
	//// !2


	//// 3. declare pointers to device arrays
	//
	//int* d_piarrState0 = 0;
	//int* d_piarrState1 = 0;
	//
	//// !3

	//// 4. allocate memory to device arrays
	//
	//cudaMalloc(&d_piarrState0, IImgrows * (IDeltaT + 1) * IImgcols * sizeof(int));
	//cudaMalloc(&d_piarrState1, IImgrows * (IDeltaT + 1) * IImgcols * sizeof(int));
	//
	//// !4
	//
	//// 5  Initialize the device arrays with zeros
	//
	//cudaMemset(d_piarrState0, 0, IImgrows * (IDeltaT + 1) * IImgcols * sizeof(int));
	//cudaMemset(d_piarrState1, 0, IImgrows * (IDeltaT + 1) / 2 * IImgcols * sizeof(int));
	//// !5

	//// 6. allocate memory to device  auxiliary arrays
	//
	//float* d_arr_val0 = 0;
	//cudaMalloc(&d_arr_val0, IImgrows / 2 * sizeof(float));

	//float* d_arr_val1 = 0;
	//cudaMalloc(&d_arr_val1, IImgrows / 2 * sizeof(float));

	//int* d_arr_deltaTLocal = 0;
	//cudaMalloc(&d_arr_deltaTLocal, IImgrows / 2 * sizeof(int));

	//int* d_arr_dT_MI = 0;
	//cudaMalloc(&d_arr_dT_MI, IImgrows * (IDeltaT + 1) * sizeof(int));

	//int* d_arr_dT_ML = 0;
	//cudaMallocManaged(&d_arr_dT_ML, IImgrows * (IDeltaT + 1) * sizeof(int));

	//int* d_arr_dT_RI = 0;
	//cudaMalloc(&d_arr_dT_RI, IImgrows * (IDeltaT + 1) * sizeof(int));

	//// 2. calculations	
	//
	//int num = 1000;
	//auto start = std::chrono::high_resolution_clock::now();

	//for (int i = 0; i < num; ++i)
	//{
	//	fncFdmt_cu_v0(piarrImage  // input image
	//		, d_piarrImage
	//		, IImgrows, IImgcols // dimensions of input image 	
	//		, d_piarrState0		// auxillary allocated buffer of mrmory in device
	//		, d_piarrState1		// auxillary allocated buffer of mrmory in device
	//		, IDeltaT
	//		, I_F
	//		, VAl_dF
	//		, d_arr_val0 		 // auxillary allocated buffer of mrmory in device
	//		, d_arr_val1			// auxillary allocated buffer of mrmory in device
	//		, d_arr_deltaTLocal	 // auxillary allocated buffer of mrmory in device
	//		, d_arr_dT_MI			// auxillary allocated buffer of mrmory in device
	//		, d_arr_dT_ML			// auxillary allocated buffer of mrmory in device
	//		, d_arr_dT_RI			// auxillary allocated buffer of mrmory in device
	//		, VAlFmin, VAlFmax, IMaxDT, u_piarrImOut);
	//}
	//
	//auto end = std::chrono::high_resolution_clock::now();	
	//auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);	
	//std::cout << "Time taken by function fncFdmt_cu_v0: " << duration.count() /((double)num)<< " microseconds" << std::endl;
	//// !2
	//
	//// output in .npy:IImgcols * IMaxDT * sizeof(int));
	//int * piarrImOut = (int*)malloc(IImgcols * IMaxDT * sizeof(int));
	//cudaMemcpy(piarrImOut, u_piarrImOut, IImgcols* IMaxDT * sizeof(int), cudaMemcpyDeviceToHost);
	//std::vector<int> v1(piarrImOut, piarrImOut + IImgcols * IMaxDT);

	//std::array<long unsigned, 2> leshape101 {IImgcols , IMaxDT};

	//npy::SaveArrayAsNumpy(strPathOutImageNpyFile_gpu, false, leshape101.size(), leshape101.data(), v1);

	//
	//
	////--------------------------------------------------------------------------------------------------------------
	////-------------------- end of calculations ------------------------------------------------------------------------------------------
	////------------------- begin to draw output image for cuda -------------------------------------------------------------------------------------------
	//
	//float flops = 0;
	//if (iImRows == 512)
	//{
	//	flops = GFLPS_512;
	//}
	//else
	//{
	//	if (iImRows == 1024)
	//	{
	//		if (BDIM_512_1024)
	//		{
	//			flops = GFLPS_512_1024;
	//		}
	//		else
	//		{
	//			flops = GFLPS_1024;
	//		}
	//	}
	//	else
	//	{
	//		flops = GFLPS_2048;
	//	}
	//}

	//cout << "GFLP/sec = " << ((double)flops) / ((double)duration.count() / ((double)num)) * 1.0e6  << endl;

	///*int deviceId;
	//int numberOfSMs;	
	//cudaGetDevice(&deviceId);
	//cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
	//cudaDeviceProp deviceProps;
	//cudaGetDeviceProperties(&deviceProps, deviceId);
	//std::string deviceName = deviceProps.name;
	//std::cout << "Device Name: " << deviceName << std::endl;
	//std::cout << "Number of SM: " << numberOfSMs << std::endl;*/



	//free(piarr);
	//cudaFree(d_piarrImage);
	//cudaFree(d_piarrState0);
	//cudaFree(d_piarrState1);
	//cudaFree(d_arr_val0);
	//cudaFree(d_arr_val1);
	//cudaFree(d_arr_deltaTLocal);
	//cudaFree(d_arr_dT_MI);
	//cudaFree(d_arr_dT_ML);
	//cudaFree(d_arr_dT_RI);
	//free(piarrImOut);

	//char filename_cpu[] = "image_cpu.png";
	//createImg_(argc, argv, v1, IImgcols, IMaxDT, filename_cpu);

	return 0;
}
template <typename T>
__global__ void vectorAdd(const T* a, const T* b, T* c, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size) {
		c[i] = a[i] + b[i];
	}
}