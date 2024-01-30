#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "FdmtCu0.cuh"

#include <math.h>
#include <stdio.h>
#include <array>
#include <iostream>
#include <string>

#include <vector>
//#include <cstdlib> // For random value generation
//#include <ctime>   // For seeding the random number generator
#include "npy.hpp"
//#include <algorithm> 
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
#include "Fragment.cuh"

//#include "Block.cuh"



//#define _CRT_SECURE_NO_WARNINGS
using namespace std;

size_t free_bytes, total_bytes;
cudaError_t cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);

extern const unsigned long long TOtal_GPU_Bytes = (long long)free_bytes;


//const char PAthGuppiFile[MAX_PATH_LENGTH] = "D://weizmann//RAW_DATA//blc20_guppi_57991_49905_DIAG_FRB121102_0011.0007.raw";
//const double VAlD_max = 3.0;
//#define LENGTH_OF_PULSE  1.0E-8


//const char PAthGuppiFile[MAX_PATH_LENGTH] = "D://weizmann//RAW_DATA//rawImit_2pow20_nchan_1npol_4_float.bin";
//const char PAthGuppiFile[MAX_PATH_LENGTH] = "D://weizmann//RAW_DATA//rawImit_2pow20_nchan_8npol_2_float.bin";
//const char PAthGuppiFile[MAX_PATH_LENGTH] = "D://weizmann//RAW_DATA//rawImit_2pow20_nchan_8npol_4_float.bin";
//const char PAthGuppiFile[MAX_PATH_LENGTH] = "D://weizmann//RAW_DATA//rawImit_2pow20_nchan_1npol_2_float.bin"; //25.0E-8
//const char PAthGuppiFile[] = "D://weizmann//RAW_DATA//rawImit_2pow20_nchan_2npol_2_float.bin";
const char PAthGuppiFile[MAX_PATH_LENGTH] = "D://weizmann//RAW_DATA//ch2.bin";//40.0E-8
//const char PAthGuppiFile[MAX_PATH_LENGTH] = "D://weizmann//RAW_DATA//ch1.bin";//  
const double VAlD_max = 1.5;
#define LENGTH_OF_PULSE  50.0E-8

const char PAthOutFile[MAX_PATH_LENGTH] = "OutPutInfo.log";

const float SIgma_Bound = 8.;
// maximal length of summation window
#define MAX_LENGTH_SUMMATION_WND 10

int main(int argc, char** argv)
{    
    CSession* pSession = new CSession(PAthGuppiFile, PAthOutFile, LENGTH_OF_PULSE, VAlD_max, SIgma_Bound, MAX_LENGTH_SUMMATION_WND);
    unsigned long long ilength = 0;
    //int iBlocks = pSession->calcQuantRemainBlocks(&ilength);
                           
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
   
    int numBlock = -1;
    int numChunk = -1;
    long long lenChunk = -1;
    int n_fdmtRows = -1, n_fdmtCols = -1;
    int sucRow = -1, sucCol = -1, width = -1;
    float cohDisp = -1., snr = -1.;
    
    CSession::read_outputlogfile_line(PAthOutFile
        , numOrder
        , &numBlock
        , &numChunk
        , &n_fdmtRows
        , &n_fdmtCols
        , &sucRow
        , &sucCol
        , &width
        , &cohDisp
        , &snr);
    //---------
    COutChunkHeader outChunkHeader(
        n_fdmtRows
        , n_fdmtCols
        , sucRow
        , sucCol
        , width
        , snr
        , cohDisp
        , numBlock-1
        , numChunk-1
    );
   
    pSession = new CSession(PAthGuppiFile, PAthOutFile, LENGTH_OF_PULSE, VAlD_max, SIgma_Bound, MAX_LENGTH_SUMMATION_WND);

    CFragment *pFRg = new CFragment();
    pSession->analyzeChunk(outChunkHeader,  pFRg);

     
    int dim = pFRg->m_dim;   
    delete pSession;

    std::array<long unsigned, 2> leshape101{ dim, dim };

    npy::SaveArrayAsNumpy("out_image.npy", false, leshape101.size(), leshape101.data(), pFRg->m_vctData);

    std::vector<float>v = pFRg->m_vctData; 
    delete pFRg;

    char filename_gpu[] = "image_gpu.png";
    createImg_(argc, argv, v, dim, dim, filename_gpu);
   
    return 0;
}

