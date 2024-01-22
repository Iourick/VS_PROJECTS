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

#include "read_and_write_log.h"
#include "Constants.h"

#include <fstream>
#include "stdio.h"
#include <sstream>
#include "Session.cuh"
#include <limits.h>
#include <stdint.h>
#include "OutChunk.h"
#include "OutChunkHeader.h"



//#define _CRT_SECURE_NO_WARNINGS
using namespace std;

size_t free_bytes, total_bytes;
cudaError_t cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);

extern const unsigned long long TOtal_GPU_Bytes = (long long)free_bytes;


const char PAthGuppiFile[MAX_PATH_LENGTH] = "D://weizmann//RAW_DATA//blc20_guppi_57991_49905_DIAG_FRB121102_0011.0007.raw";
//const char PAthGuppiFile[MAX_PATH_LENGTH] = "D://weizmann//RAW_DATA//rawImit_2pow20_float.bin";
const char PAthOutFile[MAX_PATH_LENGTH] = "OutPutInfo.log";


const double VAlD_max = 1.5;
#define LENGTH_OF_PULSE 25.0E-7//25.0E-8
const float SIgma_Bound = 5.;
// maximal length of summation window
#define MAX_LENGTH_SUMMATION_WND 10

int main()
{
    CSession* pSession = new CSession(PAthGuppiFile, PAthOutFile, LENGTH_OF_PULSE, VAlD_max, SIgma_Bound, MAX_LENGTH_SUMMATION_WND);
    unsigned long long ilength = 0;
    int iBlocks = pSession->calcQuantRemainBlocks(&ilength);
                           
    pSession->launch();

    if (pSession->m_pvctSuccessHeaders->size() > 0)
    {
        std::cout << "               Successful Chunk Numbers = " << pSession->m_pvctSuccessHeaders->size() << std::endl;
        //--------------------------------------

        char charrTemp[200] = { 0 };
        for (int i = 0; i < pSession->m_pvctSuccessHeaders->size(); ++i)
        {

            memset(charrTemp, 0, 200 * sizeof(char));
            (*(pSession->m_pvctSuccessHeaders))[i].createOutStr(charrTemp);
            std::cout << i+1<<". "<< charrTemp << std::endl;
        }
        //--------------------------------------
        pSession->writeReport();
    }

    delete pSession;

    char chInp[200] = { 0 };
    std::cout << "if you  want to quit, print q" << endl;
    std::cout << "if you want to proceed, print y " << endl;

    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    char ch0 = std::cin.get();
    if (ch0 == 'q')
    {
        return 0;
    }
    ////----------------------------------------------------------------------------------------------------
        // SECOND PART - PAINTINGS
    //----------------------------------------------------------------------------------------------------
    std::cout << "print number of chunk: " << endl;
    int numOrder = -1;
    std::cin >> numOrder;
    --numOrder;
    char strPassLog[200] = { 0 };
    strcpy(strPassLog, PAthOutFile);
    int numBlock = -1;
    int numChunk = -1;
    long long lenChunk = -1;
    CSession::read_outputlogfile_line(strPassLog
        , numOrder
        , &numBlock
        , &numChunk
        , &lenChunk);
    int lengthOfChunk = 0, quantChunks = 0;
    int arrChunks[1000] = { 0 };
    float arrCohD[1000] = { 0. };
    char strPassDataFile[200] = { 0 };

    //fncReadLog_(strPassLog, strPassDataFile, &lengthOfChunk, &quantChunks, arrChunks, arrCohD);
    //unsigned int lenarr1 = 0, n_p1 = 0;
    //float valD_max1 = 0., valf_min1 = 0., valf_max1 = 0., valSigmaBound1 = 0.;

    //if (readHeader(strPassDataFile, lenarr1, n_p1
    //    , valD_max1, valf_min1, valf_max1, valSigmaBound1) == 1)
    //{
    //    std::cerr << "Error opening file." << std::endl;
    //    return 1;
    //}
    //const int NUmChunk = arrChunks[numOrder];
    //const float VAlCohD = arrCohD[numOrder];
    //CStreamParams* pStreamPars1 = new CStreamParams(strPassDataFile, NUmChunk * lengthOfChunk, (NUmChunk + 1) * lengthOfChunk,
    //    lengthOfChunk);

    //const int iremains = pStreamPars1->m_lenarr - NUmChunk * pStreamPars->m_lenChunk;
    //if (iremains <= 0)
    //{
    //    fprintf(stderr, "Define chunk's number correctly.");
    //    return false;
    //}
    //const unsigned int LEnChunk = (iremains < pStreamPars->m_lenChunk) ? iremains : pStreamPars->m_lenChunk;

    //// create output numpy files with images

    //cudaError_t cudaStatus;
    //fdmt_type_* poutputImage = (fdmt_type_*)malloc(sizeof(fdmt_type_));

    //fdmt_type_* poutputPartImage = (fdmt_type_*)malloc(sizeof(fdmt_type_));

    //fdmt_type_** ppoutputPartImage = &poutputPartImage;
    //fdmt_type_** ppoutputImage = &poutputImage;


    //int  iargmaxCol = -1, iargmaxRow = -1;
    //fdmt_type_ valSNR = -1;
    //int quantRowsPartImage = -1;
    //createOutImageForFixedNumberChunk_gpu(ppoutputImage, &iargmaxRow, &iargmaxCol
    //    , &valSNR, ppoutputPartImage, &quantRowsPartImage, pStreamPars1, NUmChunk, VAlCohD);

    //std::cout << "OUTPUT DATA: " << endl;
    //std::cout << "CHUNK NUMBER = " << NUmChunk << endl;
    //std::cout << "SNR = " << valSNR << endl;
    //std::cout << "ROW = " << iargmaxRow << endl;
    //std::cout << "COLUMN  = " << iargmaxCol << endl;

    //std::vector<float> v1(poutputPartImage, poutputPartImage + quantRowsPartImage * quantRowsPartImage);

    //std::array<long unsigned, 2> leshape101{ quantRowsPartImage, quantRowsPartImage };

    //npy::SaveArrayAsNumpy("out_image.npy", false, leshape101.size(), leshape101.data(), v1);

    //ppoutputPartImage = nullptr;

    //free(poutputImage);
    //free(poutputPartImage);
    //delete pStreamPars1;

    //char filename_cpu[] = "image_cpu.png";
    //createImg_(argc, argv, v1, quantRowsPartImage, quantRowsPartImage, filename_cpu);

    return 0;
}

