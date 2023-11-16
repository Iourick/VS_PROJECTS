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

#define _CRT_SECURE_NO_WARNINGS
using namespace std;

class StreamParams;

void display() {
    glClear(GL_COLOR_BUFFER_BIT); // Очистка буфера цвета

    glBegin(GL_TRIANGLES); // Начало рисования треугольника
    glColor3f(1.0, 0.0, 0.0); // Установка цвета (красный)
    glVertex2f(0.0, 0.5); // Вершина 1
    glColor3f(0.0, 1.0, 0.0); // Установка цвета (зеленый)
    glVertex2f(-0.5, -0.5); // Вершина 2
    glColor3f(0.0, 0.0, 1.0); // Установка цвета (синий)
    glVertex2f(0.5, -0.5); // Вершина 3
    glEnd(); // Завершение рисования треугольника

    glFlush(); // Отправка рисунка на экран
}

int numAttemptions = 0;
int main(int argc, char** argv)
{  
    //int N = 8; // Number of data points
    //std::complex<float>* in = (std::complex<float>*)malloc(sizeof(std::complex<float>) * N);
    //std::complex<float>* out = (std::complex<float>*)malloc(sizeof(std::complex<float>) * N);
    //fftwf_complex* fftw_in = reinterpret_cast<fftwf_complex*>(in);
    //fftwf_complex* fftw_out = reinterpret_cast<fftwf_complex*>(out);
    ////fftwf_complex* out = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * N);

    //// Initialize 'in' with complex numbers using std::complex<float>
    //for (int i = 0; i < N; ++i) {
    //    in[i] = std::complex<float>(i + 1, i + 1); // Real and imaginary parts
    //}

    //// Create the FFT plan
    //fftwf_plan plan = fftwf_plan_dft_1d(N, fftw_in, fftw_out, FFTW_FORWARD, FFTW_ESTIMATE);

    //// Execute the FFT
    //fftwf_execute(plan);

    //// Output the FFT result
    //std::cout << "FFT Result fftw_out:" << std::endl;
    //for (int i = 0; i < N; ++i) {
    //    std::cout << "Element " << i << ": "
    //        << fftw_out[i][0] << " + " << fftw_out[i][1] << "i" << std::endl;
    //}

    //std::cout << "FFT Result out:" << std::endl;
    //for (int i = 0; i < N; ++i) {
    //    std::cout << "Element " << i << ": "
    //        << out[i] << std::endl;
    //}

    //// Destroy the plan and free allocated memory
    //fftwf_destroy_plan(plan);


    //std::complex<float>* in1 = (std::complex<float>*)malloc(sizeof(std::complex<float>) * N);
    //fftwf_complex* fftw_in1 = reinterpret_cast<fftwf_complex*>(in1);
    //plan = fftwf_plan_dft_1d(N, fftw_out, fftw_in1, FFTW_BACKWARD, FFTW_ESTIMATE);
    //fftwf_execute(plan);
    //std::cout << "FFT Result in1:" << std::endl;
    //for (int i = 0; i < N; ++i) {
    //    std::cout << "Element " << i << ": "
    //        << in1[i] << std::endl;
    //}
    //fftwf_destroy_plan(plan);


    ////fftw_free(fftw_out);
    //free(in);
    //free(out);
    //free(in1);

    //return 0;



    cout << "By default input file is  \"D://MyVSprojPy//hybrid//info.bin\"" << endl;
    cout << "if you want to use one of your own, enter the pass  with double quotation marks \"..\"" << endl;
    cout<<"if you don't want, print n" << endl;
    cout << "if you  want to quit, print q" << endl;

    char userInput[200];
    char chInpFilePass[200] = { 0 };
    strcpy(chInpFilePass,"D://MyVSprojPy//hybrid//info.bin");

    
    cin.getline(userInput,200);
    cout << userInput<< endl;


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
    cout << "Header's information:" << endl;
    cout << "Length of time serie = " << lenarr << endl;

    cout << "If you want go on by default print Y, otherwise print any symbol: ";
    cin.getline(userInput, 200);
    int numBegin = 0, numEnd = 0, lenChunk = 0;
    if (strcmp(userInput, "Y")!=0)
    {       

        for (int i = 0; i < 4; ++i)
        {
            cout << "Print begin number of time serie: ";
            std::cin >> numBegin;

            cout << "Print end number of time serie: ";
            std::cin >> numEnd;

            cout << "Print chunk's length: ";
            std::cin >> lenChunk;

            if ((numBegin < 1) || (numEnd > lenarr) || (lenChunk > (numEnd - numBegin)))
            {
                cout << "Check up parametres" << endl;
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

    // ZAGLUSHKA !!~!
    numBegin = 0;
    numEnd = lenarr -1;
    lenChunk = lenarr;
    // !

    CStreamParams StreamPars(chInpFilePass,  numBegin, numEnd, lenChunk);
    int* piarrNumSucessfulChunks = (int*)malloc(sizeof(int) *(1 + (numEnd - numBegin)/ lenChunk));
    float* parrCoherent_d = (float*)malloc(sizeof(float) * (1 + (numEnd - numBegin) / lenChunk));
    int quantOfSuccessfulChunks = 0;    

    const int IMaxQuantChunks = (numEnd - numBegin + lenChunk - 1) / lenChunk;  
    float* parrSuccessImagesBuff = (float*)malloc(sizeof(float) * (n_p * (lenChunk / n_p) * IMaxQuantChunks));


    int irez = fncHybridScan(parrSuccessImagesBuff, piarrNumSucessfulChunks, parrCoherent_d, quantOfSuccessfulChunks, &StreamPars);

    // create output numpy files with images
    // calc dimensions:
    int it = (StreamPars.m_lenChunk / StreamPars.m_n_p);
    float*  outputImage = (float*)malloc((StreamPars.m_n_p) * (StreamPars.m_lenChunk / StreamPars.m_n_p)
            * sizeof(float));
    float* outputPartImage = (float*)malloc((StreamPars.m_n_p) * (StreamPars.m_n_p) * sizeof(float));

    for (int i = 0; i < quantOfSuccessfulChunks; ++i)
    { 
        createOutImageForFixedNumberChunk(outputImage, &StreamPars, piarrNumSucessfulChunks[i], parrCoherent_d[i]);
        for (int j = 0; j < StreamPars.m_n_p; ++j)
        {
            memcpy(&outputPartImage[j * StreamPars.m_n_p], &parrSuccessImagesBuff[ j * (StreamPars.m_lenChunk / StreamPars.m_n_p)]
                , (StreamPars.m_n_p) * sizeof(float));
        }
        
        std::vector<float> v1(outputPartImage, outputPartImage + (StreamPars.m_n_p) * (StreamPars.m_n_p));

        std::array<long unsigned, 2> leshape101 {StreamPars.m_n_p, (StreamPars.m_n_p)};

        npy::SaveArrayAsNumpy("out_image.npy", false, leshape101.size(), leshape101.data(), v1);
        
    }     
    
    free(piarrNumSucessfulChunks);
    free(outputImage);
    free(outputPartImage);
    free(parrCoherent_d);
    free(parrSuccessImagesBuff);



    return 0;
    
}
