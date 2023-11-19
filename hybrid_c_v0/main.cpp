#include <iostream>
#include <complex>
#include <fftw3.h>
#include <GL/glut.h>
#include <fstream>
#include "HybridC_v0.h"
#include "utilites.h"
#include "StreamParams.h"

#include <array>
#include <iostream>
#include <string>

#include <vector>
#include <cstdlib> // For random value generation
#include <ctime>   // For seeding the random number generator
#include "npy.hpp"
#include <algorithm> 

#include <chrono>
#include "fileInput.h"
#include "DrawImg.h"
#include "Constants.h"
#include "read_and_write_log.h"

#define _CRT_SECURE_NO_WARNINGS
using namespace std;

class StreamParams;

//void display() {
//    glClear(GL_COLOR_BUFFER_BIT); // Очистка буфера цвета
//
//    glBegin(GL_TRIANGLES); // Начало рисования треугольника
//    glColor3f(1.0, 0.0, 0.0); // Установка цвета (красный)
//    glVertex2f(0.0, 0.5); // Вершина 1
//    glColor3f(0.0, 1.0, 0.0); // Установка цвета (зеленый)
//    glVertex2f(-0.5, -0.5); // Вершина 2
//    glColor3f(0.0, 0.0, 1.0); // Установка цвета (синий)
//    glVertex2f(0.5, -0.5); // Вершина 3
//    glEnd(); // Завершение рисования треугольника
//
//    glFlush(); // Отправка рисунка на экран
//}

int numAttemptions = 0;
int main(int argc, char** argv)
{  
    std::cout << "By default input file is  \"D://MyVSprojPy//hybrid//info.bin\"" << endl;
    std::cout << "if you want to use one of your own, enter the pass  with double quotation marks \"..\"" << endl;
    std::cout<<"if you don't want, print n" << endl;
    std::cout << "if you  want to quit, print q" << endl;

    char userInput[200];
    char chInpFilePass[200] = { 0 };
    strcpy(chInpFilePass,"D://MyVSprojPy//hybrid//info.bin");

    
    cin.getline(userInput,200);
    //std::cout << userInput<< endl;


    if (strchr(userInput, '"') != nullptr)
    {
        memcpy(chInpFilePass, &userInput[1], (strlen(userInput) - 2) * sizeof(char));
        //strcpy(chInpFilePass, userInput);       
    }
    
    unsigned int lenarr = 0, n_p = 0;
    float valD_max = 0., valf_min = 0., valf_max = 0., valSigmaBound = 0.;

    if (readHeader(chInpFilePass, lenarr, n_p
        , valD_max, valf_min, valf_max, valSigmaBound) == 1)
    {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }
    std::cout << "Header's information:" << endl;
    std::cout << "Length of time serie = " << lenarr << endl;

    std::cout << "If you want go on by default print y, otherwise print any symbol: ";
    cin.getline(userInput, 200);
    int numBegin = 0, numEnd = 0, lenChunk = 0;
    // ZAGLUSHKA !!~!
    numBegin = 0;
    numEnd = lenarr -1;
    lenChunk = pow(2,20);
    // !
    if (strcmp(userInput, "y")!=0)
    {       

        for (int i = 0; i < 4; ++i)
        {
            std::cout << "Print begin number of time serie: ";
            std::cin >> numBegin;

            std::cout << "Print end number of time serie: ";
            std::cin >> numEnd;

            std::cout << "Print chunk's length: ";
            std::cin >> lenChunk;

            if ((numBegin < 1) || (numEnd > lenarr) || (lenChunk > (numEnd - numBegin)))
            {
                std::cout << "Check up parametres" << endl;
                ++numAttemptions;
                if (numAttemptions == 4)
                {
                    return 0;
                }
            }
            else
            {
                break;
            }
        }
    }

    
    CStreamParams* pStreamPars = new CStreamParams(chInpFilePass, numBegin, numEnd, lenChunk);
    //CStreamParams StreamPars(chInpFilePass,  numBegin, numEnd, lenChunk);
    int* piarrNumSucessfulChunks = (int*)malloc(sizeof(int) *(1 + (numEnd - numBegin)/ lenChunk));
    float* parrCoherent_d = (float*)malloc(sizeof(float) * (1 + (numEnd - numBegin) / lenChunk));
    int quantOfSuccessfulChunks = 0;    

   
   

    
    //int irez = fncHybridScan(nullptr, piarrNumSucessfulChunks, parrCoherent_d, quantOfSuccessfulChunks, &StreamPars);
    int irez = fncHybridScan(nullptr, piarrNumSucessfulChunks, parrCoherent_d, quantOfSuccessfulChunks, pStreamPars);

    fncWriteLog_("info.log", chInpFilePass, "hybrid dedispersion, C++ implementation"
        , lenChunk, quantOfSuccessfulChunks, piarrNumSucessfulChunks, parrCoherent_d,0);


    
    
    std::cout << "------------ Calculations completed successfully -------------" << endl;
    std::cout << "Pass to Data File : " << chInpFilePass << endl;
    std::cout << "Successful Chunks Number : " << quantOfSuccessfulChunks<< endl;
    std::cout << "Chunk Num., Coh. Disp. : " << endl;
    for (int i = 0; i < quantOfSuccessfulChunks; ++i)
    {
        std::cout << piarrNumSucessfulChunks [i]<<" ; " << parrCoherent_d[i] << endl;
    }

    free(piarrNumSucessfulChunks);
    free(parrCoherent_d);
    //fclose(StreamPars.m_stream);
    delete pStreamPars;

    std::cout << "Running Time = " << 0. << "ms"<<endl;
    std::cout << "---------------------------------------------------------" << endl;


    
    char chInp[200] = { 0 };
    std::cout << "if you  want to quit, print q" << endl;
    std::cout << "if you want to proceed, print y " << endl;
    cin.getline(chInp, 200);
    if (chInp == "q")
    {
        return 0;
    }
    std::cout << "print number of chunk: " << endl;
    int numOrder = -1;
    cin >> numOrder;
    

    // SECOND PART



    char strPassLog[] = "info.log";
    int lengthOfChunk = 0, quantChunks = 0;
    int arrChunks[1000] = { 0 };
    float arrCohD[1000] = { 0. };
    char strPassDataFile[200] = { 0 };
    
    fncReadLog_(strPassLog, strPassDataFile, &lengthOfChunk, &quantChunks, arrChunks, arrCohD);
    unsigned int lenarr1 = 0, n_p1 = 0;
    float valD_max1 = 0., valf_min1 = 0., valf_max1 = 0., valSigmaBound1 = 0.;

    if (readHeader(strPassDataFile, lenarr1, n_p1
        , valD_max1, valf_min1, valf_max1, valSigmaBound1) == 1)
    {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }
    const int NUmChunk = arrChunks[numOrder];
    const float VAlCohD = arrCohD[numOrder];
    CStreamParams StreamPars1(strPassDataFile, NUmChunk * lengthOfChunk, (NUmChunk + 1) * lengthOfChunk,
        lengthOfChunk);
    
    // create output numpy files with images
   
    
    float*  poutputImage = (float*)malloc((StreamPars1.m_n_p) * (StreamPars1.m_lenChunk / StreamPars1.m_n_p)
            * sizeof(float));
    float* poutputPartImage = (float*)malloc( sizeof(float));
    float** ppoutputPartImage = &poutputPartImage;   


    int  iargmaxCol = -1, iargmaxRow = -1;
    float valSNR = -1;
    int quantRowsPartImage = -1;
    createOutImageForFixedNumberChunk(poutputImage,&iargmaxRow, &iargmaxCol, &valSNR, ppoutputPartImage,&quantRowsPartImage, &StreamPars1, NUmChunk, VAlCohD); 

    std::cout << "OUTPUT DATA: " << endl;
    std::cout << "CHUNK NUMBER = " << NUmChunk <<endl;
    std::cout << "SNR = " << valSNR << endl;
    std::cout << "ROW = " << iargmaxRow << endl;
    std::cout << "COLUMN  = " << iargmaxCol << endl;
        
    std::vector<float> v1(poutputPartImage, poutputPartImage + quantRowsPartImage * quantRowsPartImage);

    std::array<long unsigned, 2> leshape101 { quantRowsPartImage, quantRowsPartImage };

    npy::SaveArrayAsNumpy("out_image.npy", false, leshape101.size(), leshape101.data(), v1);    
         
    ppoutputPartImage = nullptr;
    
    free(poutputImage);
    free(poutputPartImage);
    
   

    char filename_cpu[] = "image_cpu.png";
    createImg_(argc, argv, v1, quantRowsPartImage, quantRowsPartImage, filename_cpu);

    return 0;    
}
