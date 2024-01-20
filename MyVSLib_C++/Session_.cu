#include "Session_.cuh"
#include <string>
#include "stdio.h"
#include <iostream>
#include "Block.cuh"
#include "OutChunk.h"
#include <cufft.h>
#include "fdmtU_cu.cuh"

CSession::~CSession()
{    
    if (m_rbFile)
    {
        fclose(m_rbFile);
    }
    m_rbFile = NULL;
    if (m_wb_file)
    {
        fclose(m_wb_file);
    }
    m_wb_file = NULL;

}
//-------------------------------------------
CSession::CSession()
{    
    m_rbFile = NULL;
    memset(m_strGuppiPath, 0, MAX_PATH_LENGTH * sizeof(char));
    memset(m_strOutPutPath, 0, MAX_PATH_LENGTH * sizeof(char));
    m_numSubStream = 0;   
    m_header = CGuppHeader();
    m_t_p = 1.0E-6;
    m_d_max = 0.;
    m_sigma_bound = 10.;
    m_length_sum_wnd = 10;
}

//--------------------------------------------
CSession::CSession(const  CSession& R)
{      
    if (m_rbFile)
    {
        fclose(m_rbFile);
    }
    m_rbFile = R.m_rbFile;
    memcpy(m_strGuppiPath, R.m_strGuppiPath, MAX_PATH_LENGTH * sizeof(char));
    memcpy(m_strOutPutPath, R.m_strOutPutPath, MAX_PATH_LENGTH * sizeof(char));
    m_numSubStream = R.m_numSubStream;
    m_header = R.m_header;  
    m_t_p = R.m_t_p;
    m_d_max = R.m_d_max;
    m_sigma_bound = R.m_sigma_bound;
    m_length_sum_wnd = R.m_length_sum_wnd;
}

//-------------------------------------------
CSession& CSession::operator=(const CSession& R)
{
    if (this == &R)
    {
        return *this;
    }
    
    if (m_rbFile)
    {
        fclose(m_rbFile);
    }
    m_rbFile = R.m_rbFile;
    memcpy(m_strGuppiPath, R.m_strGuppiPath, MAX_PATH_LENGTH * sizeof(char));
    memcpy(m_strOutPutPath, R.m_strOutPutPath, MAX_PATH_LENGTH * sizeof(char));
    m_numSubStream = R.m_numSubStream;
    m_header = R.m_header;    
    m_t_p = R.m_t_p;
    m_d_max = R.m_d_max;
    m_sigma_bound = R.m_sigma_bound;
    m_length_sum_wnd = R.m_length_sum_wnd;
    return *this;
}

//--------------------------------- 
CSession::CSession(const char* strGuppiPath, const char* strOutPutPath, const float t_p
,const double d_max, const float sigma_bound, const int length_sum_wnd)
{
    strcpy(m_strOutPutPath, strOutPutPath);
    strcpy(m_strGuppiPath, strGuppiPath);
    m_rbFile = fopen(strGuppiPath, "rb");    
    m_wb_file = fopen(strOutPutPath, "wb");
    m_numSubStream = 0;    
    m_t_p = t_p;
    m_d_max = d_max;
    m_sigma_bound = sigma_bound;
    m_length_sum_wnd = length_sum_wnd;
}
//------------------------------------
int CSession::calcQuantRemainBlocks(unsigned long long* pilength)
{
    const long long position = ftell(m_rbFile);
    int nbits = 0;
    float chanBW = 0;
    int npol = 0;
    bool bdirectIO = 0;
    float centfreq = 0;
    int nchan = 0;
    float obsBW = 0;
    long long nblocksize = 0;
    EN_telescope TELESCOP = GBT;
    int ireturn = 0;
    *pilength = 0;
    for (int i = 0; i < 1 << 26; ++i)
    {        
        long long pos0 = ftell(m_rbFile);
        if (!CGuppHeader::readHeader(
            m_rbFile
            , &nbits
            , &chanBW
            , &npol
            , &bdirectIO
            , &centfreq
            , &nchan
            , &obsBW
            , &nblocksize
            , &TELESCOP
        )
            )
        {
            break;
        }      

        ireturn++;
        (*pilength) += (unsigned long)nblocksize;
        unsigned long long ioffset = (unsigned long)nblocksize;
        std::cout << "i = " << i << " ; nblocksize = " << nblocksize << " (*pilength) = " << (*pilength) << std::endl;
        if (bdirectIO)
        {
            unsigned long num = (ioffset + 511) / 512;
            ioffset = num * 512;
        }

        fseek(m_rbFile, ioffset, SEEK_CUR);     
 
    }
    fseek(m_rbFile, position, SEEK_SET);
    return ireturn;
}

//---------------------------------------------------------------
int CSession::launch()
{
    // calc quantity sessions
    unsigned long long ilength = -1;

    // 1. blocks quantity calculation
    unsigned long long position = ftell(m_rbFile);
    const int IBlock = calcQuantRemainBlocks(&ilength);
    
    fseek(m_rbFile, position, SEEK_SET);
    //!1


    // 2.allocation memory for parametrs in CPU
    int nbits = 0;
    float chanBW = 0;
    int npol = 0;
    bool bdirectIO = 0;
    float centfreq = 0;
    int nchan = 0;
    float obsBW = 0;
    long long nblocksize = 0;
    EN_telescope TELESCOP = GBT;
    int ireturn = 0;
    // !2

    // 3. Performing a loop using the variable nS, nS = 0,..,IBlock. 
    //IBlock - number of bulks
    
    for (int nS = 0; nS < 1/*IBlock*/; ++nS)
    {       
        // 3.1. reading info from current bulk header
        // After return 
        // file cursor is installed on beginning of data block
        if (!CGuppHeader::readHeader(
            m_rbFile
            , &nbits
            , &chanBW
            , &npol
            , &bdirectIO
            , &centfreq
            , &nchan
            , &obsBW
            , &nblocksize
            , &TELESCOP
        )
            )
        {
            writeReport(nS);
            return -1;
        } 
        fclose(m_rbFile);
        m_rbFile = fopen("D://weizmann//RAW_DATA//rawImit_2pow20_.bin", "rb");
        nbits = sizeof(inp_type_)*8;
        int nchan = -1;
        float fmin = -1, fmax = -1.;
        fread(&nchan, sizeof(int), 1, m_rbFile);
        fread(&npol, sizeof(int), 1, m_rbFile);
        fread(&nblocksize, sizeof(int), 1, m_rbFile);
        fread(&fmin, sizeof(float), 1, m_rbFile);
        fread(&fmax, sizeof(float), 1, m_rbFile);
        chanBW = -(fmax - fmin) / ((float)nchan);
        bdirectIO = false;
        centfreq = (fmax + fmin) / 2.;
        TELESCOP = GBT;
            
        // 1!

        // 2. creating a current GuppHeader
        m_header = CGuppHeader(
            nbits
            , chanBW
            , npol
            , bdirectIO
            , centfreq
            , nchan
            , obsBW
            , nblocksize
            , TELESCOP
        );
        // 2!

        // calculate N_p
        const int len_sft = calc_len_sft(fabs(m_header.m_chanBW));

        // calculate lenChunk along time axe
        const unsigned int LenChunk = calcLenChunk(len_sft);
        //
        const bool bCHannel_order = (m_header.m_chanBW > 0.) ? true : false;
        CBlock* pBlock = new CBlock(
              m_header.m_centfreq - fabs(m_header.m_chanBW) * m_header.m_nchan / 2.
            , m_header.m_centfreq + fabs(m_header.m_chanBW) * m_header.m_nchan / 2.
            , m_header.m_npol
            , m_header.m_nblocksize
            , m_header.m_nchan            
            , LenChunk
            , len_sft
            , nS
            , m_header.m_nbits
            , bCHannel_order
            , m_d_max
            , m_sigma_bound
            , m_length_sum_wnd 

        );

        int quantSuccessChunks = 0;

        pBlock->process(m_rbFile ,&quantSuccessChunks);

        

        if (quantSuccessChunks > 0)
        {
            std::cout << "Block number = " << nS << "; Successful Chunks Number = " << quantSuccessChunks << std::endl;
        }
        delete pBlock;
        
        unsigned long long ioffset = m_header.m_nblocksize;
        
        if (bdirectIO)
        {
            unsigned long long num = (ioffset + 511) / 512;
            ioffset = num * 512;
        }

        fseek(m_rbFile, ioffset, SEEK_CUR);        


    }

    return 0;
}
//-------------------------------------------
long long CSession::calcLenChunk(const int n_p)
{
    const int N_p = n_p * m_header.m_nchan;
    unsigned int IMaxDT = n_p * m_header.m_nchan;
    float fmin = m_header.m_centfreq - fabs(m_header.m_chanBW) * m_header.m_nchan / 2.;
    float fmax = m_header.m_centfreq + fabs(m_header.m_chanBW) * m_header.m_nchan / 2.;
    const int  IDeltaT = calc_IDeltaT(N_p, fmin, fmax, IMaxDT);

    float valNominator = TOtal_GPU_Bytes - N_p * (sizeof(float) + sizeof(int)
        / 2 + 3 * (IDeltaT + 1) * sizeof(int));

    float valDenominator = m_header.m_nchan * m_header.m_npol * m_header.m_nbits / 8
        + m_header.m_nchan * m_header.m_npol / 2 * sizeof(cufftComplex)
        + sizeof(fdmt_type_) + 2 * (IDeltaT + 1) * sizeof(fdmt_type_)
        + 3 * m_header.m_nchan * m_header.m_npol * sizeof(cufftComplex)/2 + 2 * +sizeof(fdmt_type_);
    float tmax = valNominator / valDenominator;
    float treal = (float)(m_header.m_nblocksize / m_header.m_nchan / m_header.m_npol/ m_header.m_nbits*8);
    float t = (tmax < treal) ? tmax : treal;    

    return  pow(2, floor(log2(t)));
}
//-----------------------------------------------------------------
void CSession::writeReport(int nS)
{

}
