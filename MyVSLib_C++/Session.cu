#include "Session.cuh"
#include <string>
#include "stdio.h"
#include <iostream>
#include "Block.cuh"
#include "OutChunk.h"
#include <cufft.h>
#include "fdmtU_cu.cuh"
#include <stdlib.h>
 

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

    if (m_pvctSuccessHeaders)
    {
        delete m_pvctSuccessHeaders;
    }

}
//-------------------------------------------
CSession::CSession()
{    
    m_rbFile = NULL;
    memset(m_strGuppiPath, 0, MAX_PATH_LENGTH * sizeof(char));
    memset(m_strOutPutPath, 0, MAX_PATH_LENGTH * sizeof(char));
    m_pvctSuccessHeaders = new std::vector<COutChunkHeader>();
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
    if (m_pvctSuccessHeaders)
    {
        m_pvctSuccessHeaders = R.m_pvctSuccessHeaders;
    }
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
    if (m_pvctSuccessHeaders)
    {
        m_pvctSuccessHeaders = R.m_pvctSuccessHeaders;
    }
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
    m_pvctSuccessHeaders = new std::vector<COutChunkHeader>();
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
    
    for (int nS = 0; nS <  IBlock; ++nS)
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
            return -1;
        }        
      
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

        //int quantSuccessChunks = 0;

        pBlock->process(m_rbFile , m_pvctSuccessHeaders);      
                
        delete pBlock;
        
        unsigned long long ioffset = m_header.m_nblocksize;
        
        if (bdirectIO)
        {
            unsigned long long num = (ioffset + 511) / 512;
            ioffset = num * 512;
        }

        fseek(m_rbFile, ioffset, SEEK_CUR);  
    }
    //if (m_pvctSuccessHeaders->size() > 0)
    //{
    //    std::cout << "               Successful Chunk Numbers = " << m_pvctSuccessHeaders->size() << std::endl;
    //    //--------------------------------------
    //    
    //    char charrTemp[200] = { 0 };
    //    for (int i = 0; i < m_pvctSuccessHeaders->size(); ++i)
    //    {

    //        memset(charrTemp, 0, 200 * sizeof(char));
    //        (*m_pvctSuccessHeaders)[i].createOutStr(charrTemp);
    //        std::cout << charrTemp << std::endl;            
    //    }
    //    //--------------------------------------
    //    writeReport();
    //}
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
        + 3 * m_header.m_nchan * m_header.m_npol * sizeof(cufftComplex) / 2 + 2 * +sizeof(fdmt_type_);
    float tmax = valNominator / valDenominator;
    float treal = (float)(m_header.m_nblocksize / m_header.m_nchan / m_header.m_npol / m_header.m_nbits * 8);
    float t = (tmax < treal) ? tmax : treal;

    return  pow(2, floor(log2(t)));
}

//-----------------------------------------------------------------
void CSession::writeReport()
{
    char arrch[2000] = { 0 };
    char charrTemp[200] = { 0 };
    for (int i = 0; i < m_pvctSuccessHeaders->size(); ++i)
    {
        memset(charrTemp, 0, 200 * sizeof(char));
        (*m_pvctSuccessHeaders)[i].createOutStr(charrTemp);
        strcat(arrch, charrTemp);
        memset(charrTemp, 0, 200 * sizeof(char));
        sprintf(charrTemp, ", Length of pulse= %.10e", m_t_p);
        strcat(arrch, charrTemp);
        strcat(arrch, "\n");
            //createOutStr(char* pstr)
    }
    size_t elements_written = fwrite(arrch, sizeof(char), strlen(arrch), m_wb_file);
}
//-------------------------------------------------
bool CSession::read_outputlogfile_line(char *pstrPassLog
    ,const int NUmLine
    , int* pnumBlock
    , int* pnumChunk
    , long long* plenChunk    
                            )
{
    //1. download enough data
    char line[300] = { 0 };
    //fgets(strHeader, sizeof(strHeader), r_file);
    char* line_buf = NULL;
    size_t line_buf_size = 0;
    int line_count = 0;
    size_t line_size;
    FILE* fp = fopen(pstrPassLog, "r");
    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", pstrPassLog);
        return EXIT_FAILURE;
    }

    /* Get the first line of the file. */
    for (int i = 0; i < NUmLine + 1; ++i)
    {
        fgets(line, 300, fp);
    }
    
    
   

    ///* Loop through until we are done with the file. */
    //while (line_size >= 0)
    //{
    //    /* Increment our line count */
    //    line_count++;

    //    /* Show the line details */
    //    printf("line[%06d]: chars=%06zd, buf size=%06zu, contents: %s", line_count,
    //        line_size, line_buf_size, line_buf);

    //    /* Get the next line */
    //    line_size = getline(&line_buf, &line_buf_size, fp);
    //}

    //fread(line, sizeof(char), MAX_HEADER_LENGTH, r_file);
    //// !

    //2. check up mode. if mode != RAW return false  
    char* p = strstr(line, "Block=");
    if (NULL == p)
    {
        return false;
    }
    
    *pnumBlock = atoi(p + 8);

    p = strstr(line, "Chunk=");
    *pnumChunk = atoi(p + 8);
   


    int ii = 0;
    //*pnumBlock = (i_io == 0) ? false : true;
    ////4 !  

    //// 5. alignment cursors to beginning of raw data
    //ioffset += 3;
    //if ((*bdirectIO))
    //{
    //    int num = (ioffset + 511) / 512;
    //    ioffset = num * 512;
    //}
    //fseek(r_file, ioffset - MAX_HEADER_LENGTH, SEEK_CUR);

    //// 5!

    //// 6.downloading NBITS
    //pio = strstr(strHeader, "NBITS");
    //if (NULL == pio)
    //{
    //    return false;
    //}
    //*nbits = atoi(pio + 9);
    ////6 ! 

    //// 7.downloading CHAN_BW
    //pio = strstr(strHeader, "CHAN_BW");
    //if (NULL == pio)
    //{
    //    return false;
    //}
    //*chanBW = atof(pio + 9);
    ////7 ! 

    //// 8.downloading OBSFREQ
    //pio = strstr(strHeader, "OBSFREQ");
    //if (NULL == pio)
    //{
    //    return false;
    //}
    //*centfreq = atof(pio + 9);
    ////8 !

    //// 9.downloading OBSNCHAN
    //pio = strstr(strHeader, "OBSNCHAN");
    //if (NULL == pio)
    //{
    //    return false;
    //}
    //*nchan = atoi(pio + 9);
    ////9 !

    //// 10.downloading OBSNCHAN
    //pio = strstr(strHeader, "OBSBW");
    //if (NULL == pio)
    //{
    //    return false;
    //}
    //*obsBW = atof(pio + 9);
    ////10 !

    //// 11.downloading BLOCSIZE
    //pio = strstr(strHeader, "BLOCSIZE");
    //if (NULL == pio)
    //{
    //    return false;
    //}
    //*nblocksize = atoi(pio + 9);
    ////11 !    

    //// 12.downloading OBSNCHAN
    //pio = strstr(strHeader, "TELESCOP");
    //if (NULL == pio)
    //{
    //    return false;
    //}
    //pio += 9;
    //char* pt = strstr(pio, "GBT");
    //char* pt1 = NULL;
    //*TELESCOP = GBT;
    //if (NULL == pt)
    //{
    //    pt = strstr(pio, "PARKES");
    //    if (NULL == pt)
    //    {
    //        return false;
    //    }
    //    if ((pt - pio) > 20)
    //    {
    //        return false;
    //    }
    //    *TELESCOP = PARKES;
    //}
    //else
    //{
    //    if ((pt - pio) > 20)
    //    {
    //        return false;
    //    }
    //}

    ////12 !

    //// 13.downloading NPOL
    //pio = strstr(strHeader, "NPOL");
    //if (NULL == pio)
    //{
    //    return false;
    //}
    //*npol = atoi(pio + 9);
    ////13 !
    return true;
}
