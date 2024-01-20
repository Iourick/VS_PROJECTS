#include "cuda_runtime.h"
#include "device_launch_parameters.h"


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

#include "Constants.h"
#include "StreamParams.h"
#include "HybridDedispersionStream_gpu.cuh"
#include "read_and_write_log.h"
#include "Constants.h"

//#define _CRT_SECURE_NO_WARNINGS
using namespace std;

class StreamParams;


//
//const char chStrDefaultInputPass[] = "data.bin";
const char chStrDefaultInputPass[] = "data3.bin";
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
int main(int argc, char** argv){
	
	printDeviceInfo();

    ////----------------------------------------------------------------------------------------------------
            // FIRST  PART - CALCULATIONS      d_max = 1.5; 3.0 
    //--------------------------------------------------------------------------------------------------------------
//------------------- 1) prepare to work -------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
    dialIntroducing();
    // 1. Create member of class CStreamParams
    CStreamParams* pStreamPars = new CStreamParams(mchInpFilePass, mnumBegin, mnumEnd, mlenChunk);
    // 1!

    // 2. memory allocation for output information 
    int* piarrNumSucessfulChunks = (int*)malloc(sizeof(int) * (1 + (mnumEnd - mnumBegin) / mlenChunk));
    float* parrCoherent_d = (float*)malloc(sizeof(float) * (1 + (mnumEnd - mnumBegin) / mlenChunk));
    int quantOfSuccessfulChunks = 0;
    // 2 !

//--------------------------------------------------------------------------------------------------------------
//------------------- 2) calculations -------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();
    int irez = fncHybridDedispersionStream_gpu(piarrNumSucessfulChunks, parrCoherent_d, quantOfSuccessfulChunks, pStreamPars);
    auto end0 = std::chrono::high_resolution_clock::now();
    auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(end0 - start);
    std::cout << "Time : " << duration0.count() << " microseconds" << std::endl;
//--------------------------------------------------------------------------------------------------------------
//------------------- 3) writing report -------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
 
    fncWriteLog_("info.log", mchInpFilePass, "hybrid dedispersion, GPU implementation"
        , mlenChunk, quantOfSuccessfulChunks, piarrNumSucessfulChunks, parrCoherent_d, 0); 

    
    std::cout << "------------ Calculations completed successfully -------------" << endl;
    std::cout << "Pass to Data File : " << mchInpFilePass << endl;
    std::cout << "Successful Chunks Number : " << quantOfSuccessfulChunks << endl;
    std::cout << "Chunk Num., Coh. Disp. : " << endl;
    for (int i = 0; i < quantOfSuccessfulChunks; ++i)
    {
        std::cout << i + 1 << ") : " << piarrNumSucessfulChunks[i] << " ; " << parrCoherent_d[i] << endl;
    }
//--------------------------------------------------------------------------------------------------------------
//------------------- 4) free memory -------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

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
    
    return 0;
}
