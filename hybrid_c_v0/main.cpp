#include <iostream>
#include <complex>
#include <fftw3.h>
#include <GL/glut.h>
#include <fstream>
#include "HybridC_v0.h"
#include "utilites.h"
#include "StreamParams.h"
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

    
    int numBegin = 0, numEnd = 0, lenChunk = 0;
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

    // ZAGLUSHKA !!~!
    numBegin = 0;
    numEnd = lenarr;
    lenChunk = lenarr;
    // !

    CStreamParams StreamPars(chInpFilePass,  numBegin, numEnd, lenChunk);
    int* piNumSucessfulChunks = (int*)malloc(sizeof(int) *(1 + (numEnd - numBegin)/ lenChunk));
    int quantOfSuccessfulChunks = 0;
    int irez = fncHybridScan(piNumSucessfulChunks, quantOfSuccessfulChunks, &StreamPars);
        
    

    free(piNumSucessfulChunks);
    //free(pOutIm);
    return 0;






    //    // Open the binary file for reading
    //    FILE* file = fopen("D://MyVSprojPy//hybrid//info.bin", "rb");
    //    if (file == nullptr) {
    //        std::cerr << "Error opening file." << std::endl;
    //        return 1;
    //    }

    //    // Variables to store data
    //    int i0, i1;
    //    float f0, f1, f2, f3;

    //    // Read the integer variables
    //    fread(&i0, sizeof(int), 1, file);
    //    fread(&i1, sizeof(int), 1, file);

    //    // Read the float variables
    //    fread(&f0, sizeof(float), 1, file);
    //    fread(&f1, sizeof(float), 1, file);
    //    fread(&f2, sizeof(float), 1, file);
    //    fread(&f3, sizeof(float), 1, file);

    //    // Read the complex array size
    //    

    //    // Allocate an array for the complex numbers
    //    std::complex<float>* array = new std::complex<float>[i0];

    //    
    //    fread(array, sizeof(std::complex<float>), i0, file);

    //    // Close the file
    //    fclose(file);

    //    // Use the variables as needed
    //    // ...

    //    // Don't forget to delete the array when you're done with it
    //        delete[] array;

    //    

    //int N = 8; // Number of data points
    //std::complex<float>* in = (std::complex<float>*)malloc(sizeof(std::complex<float>) * N);
    //fftwf_complex* fftw_in = reinterpret_cast<fftwf_complex*>(in);
    //fftwf_complex* out = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * N);

    //// Initialize 'in' with complex numbers using std::complex<float>
    //for (int i = 0; i < N; ++i) {
    //    in[i] = std::complex<float>(i + 1, i + 1); // Real and imaginary parts
    //}

    //// Create the FFT plan
    //fftwf_plan plan = fftwf_plan_dft_1d(N, fftw_in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    //// Execute the FFT
    //fftwf_execute(plan);

    //// Output the FFT result
    //std::cout << "FFT Result:" << std::endl;
    //for (int i = 0; i < N; ++i) {
    //    std::cout << "Element " << i << ": "
    //        << out[i][0] << " + " << out[i][1] << "i" << std::endl;
    //}

    //// Destroy the plan and free allocated memory
    //fftwf_destroy_plan(plan);
    //fftw_free(out);
    //free(in);   

  
        //glutInit(&argc, argv); // Инициализация GLUT
        //glutInitDisplayMode(GLUT_RGB); // Установка режима отображения (RGB)
        //glutInitWindowSize(800, 600); // Установка размеров окна
        //glutCreateWindow("Простой пример GLUT"); // Создание окна

        //glutDisplayFunc(display); // Установка функции отображения

        //glutMainLoop(); // Запуск главного цикла GLUT  
    
}
