
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <stdio.h>
#include <array>
#include <iostream>
#include <string>

#include <GL/glut.h>
#include <GL/gl.h>
#include <IL/il.h>
#include <IL/ilu.h>
#include <IL/ilut.h>
#include <vector>
#include <cstdlib> // For random value generation
#include <ctime>   // For seeding the random number generator
#include "npy.hpp"
#include <algorithm> 
#include "kernel.cuh"
#include <chrono>



using namespace std;
std::vector<std::vector<int>> ivctOut;
int IROWS;
int ICOLS;

void display() {
    //glClearColor(0.0, 0.0, 0.0, 0.0); // Background color (black)
    glClearColor(1.0, 1.0, 1.0, 0.0); // Background color (white)

    glClear(GL_COLOR_BUFFER_BIT);

    // Set up the view
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, ICOLS, 0, IROWS);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Render the piarrOut array as an image here
    glBegin(GL_POINTS);
    for (int i = 0; i < IROWS; i++) {
        for (int j = 0; j < ICOLS; j++) {
            float grayscale = static_cast<float>(ivctOut[i][j]) / 255.0;
            glColor3f(grayscale, grayscale, grayscale); // Grayscale color
            glVertex2f(j, i); // Draw a point for each pixel
        }
    }
    glEnd();

    glFlush();
}
//--------------------------------------------------------------------------------------

void saveImage(const char* filename) {
    ilInit();
    ilutRenderer(ILUT_OPENGL);
    ilEnable(IL_FILE_OVERWRITE);

    ILuint imageID = ilGenImage();
    ilBindImage(imageID);

    ILenum format = IL_LUMINANCE;
    ILenum type = IL_UNSIGNED_BYTE;

    std::vector<ILubyte> pixelData(IROWS * ICOLS);

    for (int i = 0; i < IROWS; i++) {
        for (int j = 0; j < ICOLS; j++) {
            pixelData[i * ICOLS + j] = static_cast<ILubyte>(ivctOut[i][j]);
        }
    }

    ilTexImage(ICOLS, IROWS, 1, 1, format, type, pixelData.data());
    ilSave(IL_PNG, filename);

    ilDeleteImages(1, &imageID);
}

//--------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
	//const std::vector<int> data1 {1, 2, 3, 4, 5, 6};
	//std::array<long unsigned, 2> leshape11 {2, 3};
	//std::array<long unsigned, 1> leshape12 {6};

	//const double data2[]{ 7 };
	//std::array<long unsigned, 3> leshape21 {1, 1, 1};
	//std::array<long unsigned, 0> leshape22 {};

	//const std::array<double, 0> data3;
	//std::array<long unsigned, 2> leshape31 {4, 0};

	//npy::SaveArrayAsNumpy("out11.npy", false, leshape11.size(), leshape11.data(), data1);
	//npy::SaveArrayAsNumpy("out12.npy", false, leshape12.size(), leshape12.data(), data1);

	//npy::SaveArrayAsNumpy("out21.npy", false, leshape21.size(), leshape21.data(), data2);
	//npy::SaveArrayAsNumpy("out22.npy", false, leshape22.size(), leshape22.data(), data2);

	//npy::SaveArrayAsNumpy("out31.npy", false, leshape31.size(), leshape31.data(), data3.data());

	//std::vector<unsigned long> sh {};
	//std::vector<int> vctD;
	//bool bf = false;
	//npy::LoadArrayFromNumpy("out12.npy", sh, bf, vctD);

	//int n = 5;  // Specify the length of the array
	//int iarr[] = { 1, 2, 3, 4, 9 };  // Your one-dimensional integer array

	//// Create a std::vector from the integer array
	//std::vector<int> v(iarr, iarr + n);
	//-------------------------------------------

	bool fortran_order = false;
	// 1. loading typeofdata
	std::vector<unsigned long> shape {};
	std::vector<int> ivctDataType_maxDT;
	npy::LoadArrayFromNumpy("D://MyVSprojPy//iarrDataType_maxDT.npy", shape, fortran_order, ivctDataType_maxDT);
	const int IMaxDT = ivctDataType_maxDT[1];
	// !1

	// 2. loading XX
	std::vector<unsigned long> shape1 {};
	std::vector<int> vctXX;
	npy::LoadArrayFromNumpy("D://MyVSprojPy//XX.npy", shape1, fortran_order, vctXX);
	// !2

	// 3. loading shape
	std::vector<unsigned long> shape0 {};
	std::vector<int> ivctImShape;
	npy::LoadArrayFromNumpy("D://MyVSprojPy//iarrShape.npy", shape0, fortran_order, ivctImShape);
	// !3

	// 4. loading fmin and fmax  
	std::vector<unsigned long> shape2 {};
	std::vector<float> vctfmin_max;
	npy::LoadArrayFromNumpy("D://MyVSprojPy//fmin_max.npy", shape2, fortran_order, vctfmin_max);
	// ! 4

	// 5. creating dynamic array arrXX
	int* piarrImage = new int[vctXX.size()];
	for (int i = 0; i < vctXX.size(); ++i)
	{
		piarrImage[i] =  vctXX[i];
		
	}

	// output in .npy:
	
	
	std::vector<int> v(piarrImage, piarrImage + vctXX.size());

	std::array<long unsigned, 1> leshape126 {vctXX.size()};

	npy::SaveArrayAsNumpy("XX00.npy", false, leshape126.size(), leshape126.data(), v);
	
	// !5

	// checking dimensions
	bool bDim0_OK = false, bDim1_OK = false;
	int numcur = 2;
	for (int i = 1; i < 31; ++i)
	{
		numcur = 2 * numcur;
		if (numcur == ivctImShape[0])
		{
			bDim0_OK = true;
		}
		if (numcur == ivctImShape[1])
		{
			bDim1_OK = true;
		}
	}

	if (!(bDim0_OK && bDim1_OK))
	{
		cout << "Input dimensions must be a power of 2 " << std::endl;
		return;
	}

	// 6. allocate memory for device array

	int* piarrImOut = new int[ivctImShape[1] * IMaxDT];
	// !6

	// 7. calculations	
	//clock_t start = clock();	
	auto start = std::chrono::high_resolution_clock::now();

	fncFdmt_cu_v0(piarrImage, ivctImShape[0], ivctImShape[1]
		, vctfmin_max[0], vctfmin_max[1], ivctDataType_maxDT[1], piarrImOut);
	/*clock_t end = clock();	
	double duration = double(end - start) / CLOCKS_PER_SEC;	
	std::cout << "Time taken by function fncFdmt_cu_v0: " << duration << " seconds" << std::endl;*/
	auto end = std::chrono::high_resolution_clock::now();

    // Вычисляем разницу времени
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Выводим время выполнения в микросекундах
    std::cout << "Time taken by function fncFdmt_cu_v0: " << duration.count() << " milliseconds" << std::endl;
	// !7
	// output in .npy:
	
	std::vector<int> v1(piarrImOut, piarrImOut + ivctImShape[1] * ivctDataType_maxDT[1]);

	std::array<long unsigned, 1> leshape101 {ivctImShape[1] * ivctDataType_maxDT[1]};

	npy::SaveArrayAsNumpy("out_arr.npy", false, leshape101.size(), leshape101.data(), v1);
	
//-------------------------------------------------------------------------------
//------  otput image -------------------------------------------------------------------------
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
	IROWS = ivctImShape[1];
	ICOLS = ivctDataType_maxDT[1];
	// zaglushka
	//memcpy(piarrImOut, piarrImage, IROWS * ICOLS * sizeof(int));
	// !zaglushka
	int* pi = new int[ICOLS];
	
	int num = IROWS / 2;
	
	for (int i = 0; i < num; ++i)
	{
		memcpy(pi, &piarrImOut[i * ICOLS], ICOLS * sizeof(int));
		memcpy(&piarrImOut[i * ICOLS], &piarrImOut[(IROWS - 1- i) * ICOLS], ICOLS * sizeof(int));
		memcpy(&piarrImOut[(IROWS -1 - i) * ICOLS], pi, ICOLS * sizeof(int));
	}
	delete pi;
	int imax = *std::max_element(piarrImage, piarrImage + ivctImShape[0] * ivctImShape[1]);
	int imin = *std::min_element(piarrImage, piarrImage + ivctImShape[0] * ivctImShape[1]);
	float coeff = 255. / (double(imax));
    ivctOut = std::vector<std::vector<int>>(IROWS, std::vector<int>(ICOLS, 0)); // Initialize with your data
    for (int i = 0; i < IROWS; ++i)
    for (int  j = 0; j < ICOLS; ++j)
    {
		ivctOut[i][j] = (int)(coeff * piarrImOut[i * ICOLS + j]);
		ivctOut[i][j] = (int)piarrImOut[i * ICOLS + j];
        
    }
	
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(ICOLS, IROWS);
    glutCreateWindow("Image Viewer");

    glutDisplayFunc(display);
    // After displaying the image, save it as "image.png"
    saveImage("image.png");

    // Set up other GLUT callbacks as needed (e.g., keyboard input)

    glutMainLoop();

    
	delete piarrImage;
	delete piarrImOut;
	
    return 0;
}
//-------------------------------------------------------------------------
void fncFdmt_cu_v0(int* piarrImgInp, const int IImgrows, const int IImgcols
	, const float VAlFmin, const  float VAlFmax, const int IMaxDT, int* piarrImgOut)
{
	// 1. quant iteration's calculation
	const int I_F = (int)(log2((double)(IImgrows)));
	// !1

	const float val_dF = (VAlFmax - VAlFmin) / ((float)(IImgrows));	
	int ideltaT = int(ceil((IMaxDT - 1.) * (1. / (VAlFmin * VAlFmin)
		- 1. / ((VAlFmin + val_dF)* (VAlFmin + val_dF)))
		/ (1. / (VAlFmin
				* VAlFmin) - 1. / (VAlFmax * VAlFmax))));

	//// 2. calculation arrays with dimensions of output arrays for each iteration	
	//vector<int> ivctOutDim0(I_F + 1);
	//vector<int> ivctOutDim1(I_F + 1);
	//vector<int> ivctOutDim2(I_F + 1);
	//ivctOutDim0[0] = IImgrows;
	//ivctOutDim1[0] = ideltaT + 1;
	//ivctOutDim2[0] = IImgcols;

	//fncCalcDimensionsOfOutputArrays(&ivctOutDim0, &ivctOutDim1
	//	, &ivctOutDim2, IImgrows, ideltaT + 1
	//	, IImgcols, IMaxDT, VAlFmin
	//	, VAlFmax);
	//vector <int>mult(I_F + 1);
	//for (int i = 0; i < mult.size(); ++i)
	//{
	//	mult[i] = ivctOutDim0[i] * ivctOutDim1[i] * ivctOutDim2[i];
	//}
	//
	//auto maxElement1 = std::max_element(mult.begin(), mult.end());
	//int maxValue = *maxElement1;

	
	// 1. declare pointers to device arrays
	int* d_p0 = 0;
	int* d_p1 = 0;
	int* d_piarrOut_0 = 0;
	int* d_piarrOut_1 = 0;
	int* d_piarrImgInp = 0;
	// !1
		
	// 2. allocate memory to device arrays
	clock_t start = clock();
	cudaMalloc(&d_piarrOut_0, IImgrows * (ideltaT + 1) * IImgcols * sizeof(int));
	cudaMalloc(&d_piarrOut_1, IImgrows * (ideltaT + 1) * IImgcols * sizeof(int));
	cudaMalloc(&d_piarrImgInp, IImgrows  * IImgcols * sizeof(int));
	// !2
	clock_t end = clock();
	double duration = double(end - start) / CLOCKS_PER_SEC;
	std::cout << "Time taken by cudaMalloc: " << duration << " seconds" << std::endl;

	// 3  Initialize the device arrays with zeros
	start = clock();
	cudaMemset(d_piarrOut_0, 0, IImgrows * (ideltaT + 1) * IImgcols * sizeof(int));
	cudaMemset(d_piarrOut_1, 0, IImgrows * (ideltaT + 1) * IImgcols * sizeof(int));
	// !3
	end = clock();
	duration = double(end - start) / CLOCKS_PER_SEC;
	std::cout << "Time taken by cudaMemset: " << duration << " seconds" << std::endl;

	// 4.copy input data from host to device
	start = clock();
	cudaMemcpy(d_piarrImgInp, piarrImgInp, IImgrows * IImgcols * sizeof(int)
		, cudaMemcpyHostToDevice);
	end = clock();
	duration = double(end - start) / CLOCKS_PER_SEC;
	std::cout << "Time taken by cudaMemcpy: " << duration << " seconds" << std::endl;
	// !4

	// 5. call initialization func
	fnc_init(d_piarrImgInp, IImgrows, IImgcols, ideltaT, d_piarrOut_0);

	// output in .npy:
	int* parrinit = (int*)malloc(IImgrows * IImgcols * (1 + ideltaT) * sizeof(int));
	cudaMemcpy(parrinit, d_piarrOut_0, IImgrows * IImgcols * (1 + ideltaT) * sizeof(int)
		, cudaMemcpyDeviceToHost);
	std::vector<int> v(parrinit, parrinit + IImgrows * IImgcols * (1 + ideltaT));

	std::array<long unsigned, 1> leshape12 {IImgrows* IImgcols* (1 + ideltaT)};
	
	npy::SaveArrayAsNumpy("init_arr.npy", false, leshape12.size(), leshape12.data(), v);
	free(parrinit);
	//! 5
	
	

	// 7.pointers initialization
	d_p0 = d_piarrOut_0;
	d_p1 = d_piarrOut_1;
	// 7!

	// 8. allocate memory to device  auxiliary arrays
	start = clock();
	float* d_arr_val0 = 0;
	cudaMalloc(&d_arr_val0, IImgrows / 2 * sizeof(float));

	float* d_arr_val1 = 0;
	cudaMalloc(&d_arr_val1, IImgrows / 2 * sizeof(float));

	int* d_arr_deltaTLocal = 0;
	cudaMalloc(&d_arr_deltaTLocal, IImgrows / 2 * sizeof(int));

	// 11. memory allocation for 3 auxillaty 2 dimensional arrays on GPU
	int* d_arr_dT_MI = 0;
	cudaMalloc(&d_arr_dT_MI, IImgrows* (ideltaT + 1) * sizeof(int));

	int* d_arr_dT_ML = 0;
	cudaMallocManaged(&d_arr_dT_ML, IImgrows * (ideltaT + 1) * sizeof(int));

	int* d_arr_dT_RI = 0;
	cudaMalloc(&d_arr_dT_RI, IImgrows * (ideltaT + 1) * sizeof(int));
	
	end = clock();
	duration = double(end - start) / CLOCKS_PER_SEC;
	std::cout << "Time taken by allocate memory: " << duration << " seconds" << std::endl;
	// !8
	int iInp0 = IImgrows;
	int iInp1 = ideltaT + 1;
	
	int iOut0 = 0, iOut1 = 0, iOut2 = 0;

	// output in .npy:
	int* parrinit2 = (int*)malloc(IImgrows * IImgcols * (1 + ideltaT) * sizeof(int));
	cudaMemcpy(parrinit2, d_piarrOut_0, IImgrows* IImgcols* (1 + ideltaT) * sizeof(int)
		, cudaMemcpyDeviceToHost);
	std::vector<int> v2(parrinit2, parrinit2 + IImgrows * IImgcols * (1 + ideltaT));

	std::array<long unsigned, 1> leshape122 {IImgrows* IImgcols* (1 + ideltaT)};

	npy::SaveArrayAsNumpy("init_arr2.npy", false, leshape122.size(), leshape122.data(), v2);
	free(parrinit2);
	for (int iit = 1; iit < (I_F + 1); ++iit)
	{		
		fncFdmtIteration(d_p0, val_dF,iInp0, iInp1
			, IImgcols,  IMaxDT,  VAlFmin
			, VAlFmax, iit,  d_arr_val0
			, d_arr_val1,  d_arr_deltaTLocal
			, d_arr_dT_MI, d_arr_dT_ML, d_arr_dT_RI
			, d_p1, iOut0, iOut1);
		if (iit == 8)
		{
			int* parrinit0 = (int*)malloc(iOut0 * iOut1 * IImgcols * sizeof(int));
			cudaMemcpy(parrinit0, d_piarrOut_1, iOut0 * iOut1 * IImgcols * sizeof(int)
				, cudaMemcpyDeviceToHost);
			std::vector<int> v0(parrinit0, parrinit0 + iOut0 * iOut1 * IImgcols);

			std::array<long unsigned, 1> leshape120 {iOut0* iOut1* IImgcols};

			npy::SaveArrayAsNumpy("arr_InterimOut.npy", false, leshape120.size(), leshape120.data(), v0);
			free(parrinit0);
		}
		// exchange order of pointers
		int* d_pt = d_p0;
		d_p0 = d_p1;
		d_p1 = d_pt;
		iInp0 = iOut0;
		iInp1 = iOut1;
		
		// !
	}

	cudaMemcpy(piarrImgOut, d_p0, IImgcols * IMaxDT
		* sizeof(int), cudaMemcpyDeviceToHost);
	start = clock();
	cudaFree(d_arr_val0);
	cudaFree(d_arr_val1);
	cudaFree(d_arr_deltaTLocal);
	
	cudaFree(d_piarrOut_0);
	cudaFree(d_piarrOut_1);
	cudaFree(d_piarrImgInp);

	// 14. free memory
	cudaFree(d_arr_dT_MI);
	cudaFree(d_arr_dT_ML);
	cudaFree(d_arr_dT_RI);
	cudaDeviceReset();
	
	end = clock();
	duration = double(end - start) / CLOCKS_PER_SEC;
	std::cout << "Time taken by cudaFree: " << duration << " seconds" << std::endl;

}


//--------------------------------------------------------------------------------------
//    Input :
//    Input - 3d array, with dimensions[N_f, N_d0, Nt]
//    f_min, f_max - are the base - band begin and end frequencies.
//    The frequencies can be entered in both MHz and GHz, units are factored out in all uses.
//    maxDT - the maximal delay(in time bins) of the maximal dispersion.
//    Appears in the paper as N_{\Delta}
//A typical input is maxDT = N_f
//dataType - To naively use FFT, one must use floating point types.
//Due to casting, use either complex64 or complex128.
//iteration num - Algorithm works in log2(Nf) iterations, each iteration changes all the sizes(like in FFT)
//Output:
//3d array, with dimensions[N_f / 2, N_d1, Nt]
//    where N_d1 is the maximal number of bins the dispersion curve travels at one output frequency band
//
//    For details, see algorithm 1 in Zackay & Ofek(2014)
// F,T = Image.shape 
// d_piarrInp имеет  размерности IDim0, IDim1,IDim2
// IDim0: this is iImgrows - quantity of rows of input power image, this is F
// IDim1: changes 
// IDim2: this is iImgcols - quantity of cols of input power image, this is T 
void fncFdmtIteration(int* d_piarrInp,const float val_dF,  const int IDim0, const int IDim1
	, const int IDim2, const int IMaxDT, const float VAlFmin
	, const float VAlFmax, const int ITerNum, float* d_arr_val0
	, float* d_arr_val1, int* d_iarr_deltaTLocal
	, int* d_iarr_dT_MI,  int* d_iarr_dT_ML,  int* d_iarr_dT_RI
	, int* d_piarrOut, int & iOutPutDim0, int& iOutPutDim1)
{
	// output in .npy:
	int* parrinit3 = (int*)malloc(IDim0 * IDim1 * IDim2 * sizeof(int));
	cudaMemcpy(parrinit3, d_piarrInp, IDim0 * IDim1 * IDim2 * sizeof(int)
		, cudaMemcpyDeviceToHost);
	std::vector<int> v3(parrinit3, parrinit3 + IDim0 * IDim1 * IDim2);

	std::array<long unsigned, 1> leshape122 {IDim0* IDim1* IDim2};

	npy::SaveArrayAsNumpy("init_arr3.npy", false, leshape122.size(), leshape122.data(), v3);
	free(parrinit3);
	float valDeltaF = pow(2., ITerNum) * val_dF;
	float temp0 = 1. / (VAlFmin * VAlFmin) -
		1. / ((VAlFmin + valDeltaF) * (VAlFmin + valDeltaF));

	const float VAlTemp1 = 1. / (VAlFmin * VAlFmin) - 1. / (VAlFmax * VAlFmax);

	int ideltaT = (int)(ceil(((float)IMaxDT - 1.0) * temp0 / VAlTemp1));
	iOutPutDim1 = ideltaT + 1;
	iOutPutDim0 = IDim0 / 2;
		

	// set zeros in output array
	cudaMemset(d_piarrOut, 0, iOutPutDim0 * iOutPutDim1 * IDim2 * sizeof(int));
	// !
	 
	float val_correction = 0;
	if (ITerNum > 0)
	{
		val_correction = val_dF / 2.;
	}

	
	// 9. auxiliary constants initialization
	const float VAlC2 = (VAlFmax - VAlFmin) / ((float)(iOutPutDim0));
	// !9	

	// 10. calculating first 3 auxillary 1 dim arrays
	int threadsPerBlock = 1024;
	int numberOfBlocks = (iOutPutDim0 + threadsPerBlock - 1) / threadsPerBlock;
	int F_Jumps = iOutPutDim0;
	float* arr_val0 = new float[F_Jumps];
	float* arr_val1 = new float[F_Jumps];
	int* iarr_deltaTLocal = new int[F_Jumps];
	for (int i_F = 0; i_F < F_Jumps; ++i_F)
	{
		
		float valf_start = VAlC2 * i_F + VAlFmin;
		float valf_end = VAlC2 * ( 1. + i_F) + VAlFmin;
		
		float valf_middle = (valf_end - valf_start)/2. + valf_start - val_correction;
		float valf_middle_larger = (valf_end - valf_start) / 2. + valf_start + val_correction;
		float temp0 = 1. / (valf_start * valf_start) - 1. / (valf_end * valf_end);

		arr_val0[i_F] = -(1. / (valf_middle * valf_middle) - 1. / (valf_start * valf_start)) / temp0;

		arr_val1[i_F] = -(1. / (valf_middle_larger * valf_middle_larger)
			- 1. / (valf_start * valf_start)) / temp0;

		iarr_deltaTLocal[i_F] = (int)(ceil((((float)(IMaxDT)) - 1.) * temp0 / VAlTemp1));
		int iii = 0;
	}
	cudaMemcpy(d_arr_val0, arr_val0, iOutPutDim0 * sizeof(float)
		, cudaMemcpyHostToDevice);
	cudaMemcpy(d_arr_val1, arr_val1, iOutPutDim0 * sizeof(float)
		, cudaMemcpyHostToDevice);
	cudaMemcpy(d_iarr_deltaTLocal, iarr_deltaTLocal, iOutPutDim0 * sizeof(int)
		, cudaMemcpyHostToDevice);
	/*# calc 3 massives of parametres
		temp1 = (1. / f_min * *2 - 1. / f_max * *2)
		for i_F in range(F_jumps) :
			f_start = (f_max - f_min) / float(F_jumps) * (i_F)+f_min
			f_end = (f_max - f_min) / float(F_jumps) * (i_F + 1) + f_min
			f_middle = (f_end - f_start) / 2. + f_start - correction
			# it turned out in the end, that putting the correction + dF to f_middle_larger(or -dF / 2 to f_middle, and +dF / 2 to f_middle larger)
			# is less sensitive than doing nothing when dedispersing a coherently dispersed pulse.
			# The confusing part is that the hitting efficiency is better with the corrections(!? !).
			f_middle_larger = (f_end - f_start) / 2 + f_start + correction
			temp0 = (1. / f_start * *2 - 1. / (f_end) * *2)
			arr_val0[i_F] = -(1. / f_middle * *2 - 1. / f_start * *2) / temp0
			arr_val1[i_F] = -(1. / f_middle_larger * *2 - 1. / f_start * *2) / temp0
			arr_deltaTLocal[i_F] = int(np.ceil((maxDT - 1) * temp0 / temp1))*/
	/*create_auxillary_1d_arrays << <numberOfBlocks, threadsPerBlock >> > (iOutPutDim0
		, IMaxDT, VAlTemp1, VAlC2, VAlFmin, val_correction
		, d_arr_val0, d_arr_val1, d_iarr_deltaTLocal);
	cudaDeviceSynchronize();*/
	if (4 == ITerNum)
	{
		// output in .npy:
		float* parr0 = (float*)malloc(iOutPutDim0 * sizeof(float));
		cudaMemcpy(parr0, d_arr_val0, iOutPutDim0 * sizeof(float)
			, cudaMemcpyDeviceToHost);
		std::vector<float> v1d0(parr0, parr0 + iOutPutDim0);

		std::array<long unsigned, 1> leshape000 {iOutPutDim0};

		npy::SaveArrayAsNumpy("parr0.npy", false, leshape000.size(), leshape000.data(), v1d0);



		cudaMemcpy(parr0, d_arr_val1, iOutPutDim0 * sizeof(float)
			, cudaMemcpyDeviceToHost);
		std::vector<float> v1d1(parr0, parr0 + iOutPutDim0);
		npy::SaveArrayAsNumpy("parr1.npy", false, leshape000.size(), leshape000.data(), v1d1);
		free(parr0);

		int* iarr_deltaTLocal = (int*)malloc(iOutPutDim0 * sizeof(int));
		cudaMemcpy(iarr_deltaTLocal, d_iarr_deltaTLocal, iOutPutDim0 * sizeof(int)
			, cudaMemcpyDeviceToHost);
		std::vector<int> v00(iarr_deltaTLocal, iarr_deltaTLocal + iOutPutDim0);



		npy::SaveArrayAsNumpy("iarr_deltaTLocal.npy", false, leshape000.size(), leshape000.data(), v00);
		free(iarr_deltaTLocal);
	}


	


	// output in .npy:
	int* parrinit4 = (int*)malloc(IDim0 * IDim1 * IDim2 * sizeof(int));
	cudaMemcpy(parrinit4, d_piarrInp, IDim0 * IDim1 * IDim2 * sizeof(int)
		, cudaMemcpyDeviceToHost);
	std::vector<int> v4(parrinit4, parrinit4 + IDim0 * IDim1 * IDim2);

	std::array<long unsigned, 1> leshape124 {IDim0* IDim1* IDim2};

	npy::SaveArrayAsNumpy("init_arr4.npy", false, leshape124.size(), leshape124.data(), v4);
	free(parrinit4);
	/*int* parrinit0 = (int*)malloc(iOutPutDim0  * sizeof(int));
	cudaMemcpy(parrinit0, d_iarr_deltaTLocal, iOutPutDim0  * sizeof(int)
		, cudaMemcpyDeviceToHost);
	std::vector<int> v0(parrinit0, parrinit0 + iOutPutDim0 );

	std::array<long unsigned, 1> leshape120 {iOutPutDim0* iOutPutDim1};

	npy::SaveArrayAsNumpy("arr_deltaTLocal.npy", false, leshape120.size(), leshape120.data(), v0);
	free(parrinit0);*/
	// !10

	

	// 12. calculating second 3 auxillary 2 dim arrays
	int quantEl = iOutPutDim0 * iOutPutDim1;
	threadsPerBlock = 256;
	numberOfBlocks = (quantEl + threadsPerBlock - 1) / threadsPerBlock;	

	kernel_2d_arrays << < numberOfBlocks, threadsPerBlock >> > (iOutPutDim0
		, iOutPutDim1, d_arr_val0, d_arr_val1, d_iarr_deltaTLocal
		, d_iarr_dT_MI, d_iarr_dT_ML
		, d_iarr_dT_RI);
	cudaDeviceSynchronize();
	// output in .npy:
	int* parrinit5 = (int*)malloc(IDim0 * IDim1 * IDim2 * sizeof(int));
	cudaMemcpy(parrinit5, d_piarrInp, IDim0 * IDim1 * IDim2 * sizeof(int)
		, cudaMemcpyDeviceToHost);
	std::vector<int> v5(parrinit5, parrinit5 + IDim0 * IDim1 * IDim2);

	std::array<long unsigned, 1> leshape125 {IDim0* IDim1* IDim2};

	npy::SaveArrayAsNumpy("init_arr5.npy", false, leshape125.size(), leshape125.data(), v5);
	free(parrinit5);
	if (ITerNum == 8)
	{
		// output in .npy:
		int* parrinit = (int*)malloc(iOutPutDim0 * iOutPutDim1 * sizeof(int));
		cudaMemcpy(parrinit, d_iarr_dT_MI, iOutPutDim0 * iOutPutDim1 * sizeof(int)
			, cudaMemcpyDeviceToHost);
		std::vector<int> v(parrinit, parrinit + iOutPutDim0 * iOutPutDim1);

		std::array<long unsigned, 1> leshape12 {iOutPutDim0* iOutPutDim1};

		npy::SaveArrayAsNumpy("iarr_dT_MI.npy", false, leshape12.size(), leshape12.data(), v);
		free(parrinit);
		//
		int* parrinit1 = (int*)malloc(iOutPutDim0 * iOutPutDim1 * sizeof(int));
		cudaMemcpy(parrinit1, d_iarr_dT_ML, iOutPutDim0 * iOutPutDim1 * sizeof(int)
			, cudaMemcpyDeviceToHost);
		std::vector<int> v1(parrinit1, parrinit1 + iOutPutDim0 * iOutPutDim1);

		npy::SaveArrayAsNumpy("iarr_dT_ML.npy", false, leshape12.size(), leshape12.data(), v1);
		free(parrinit1);
		//
		int* parrinit2 = (int*)malloc(iOutPutDim0 * iOutPutDim1 * sizeof(int));
		cudaMemcpy(parrinit2, d_iarr_dT_RI, iOutPutDim0 * iOutPutDim1 * sizeof(int)
			, cudaMemcpyDeviceToHost);
		std::vector<int> v2(parrinit2, parrinit2 + iOutPutDim0 * iOutPutDim1);

		npy::SaveArrayAsNumpy("iarr_dT_RI.npy", false, leshape12.size(), leshape12.data(), v2);
		free(parrinit2);


		//// output in .npy:
		//int* parrinit6 = (int*)malloc(IDim0 * IDim1 * IDim2 * sizeof(int));
		//cudaMemcpy(parrinit6, d_piarrInp, IDim0 * IDim1 * IDim2 * sizeof(int)
		//	, cudaMemcpyDeviceToHost);
		//std::vector<int> v6(parrinit5, parrinit6 + IDim0 * IDim1 * IDim2);

		//std::array<long unsigned, 1> leshape126 {IDim0* IDim1* IDim2};

		//npy::SaveArrayAsNumpy("init_arr6.npy", false, leshape126.size(), leshape126.data(), v3);
		//free(parrinit6);
	}

	// !11

    // 13. 
	quantEl =  iOutPutDim0* iOutPutDim1* IDim2;
	numberOfBlocks = (quantEl + threadsPerBlock - 1) / threadsPerBlock;
	kernel_shift << <numberOfBlocks, threadsPerBlock >> > (d_piarrInp
		, IDim0, IDim1, IDim2, d_iarr_deltaTLocal, d_iarr_dT_MI, d_iarr_dT_ML, d_iarr_dT_RI
		, iOutPutDim0, iOutPutDim1, d_piarrOut);
	/*shift_and_sum (d_piarrInp
		, IDim0, IDim1, IDim2, d_iarr_deltaTLocal, d_iarr_dT_MI, d_iarr_dT_ML, d_iarr_dT_RI
		, iOutPutDim0, iOutPutDim1, d_piarrOut);*/
	cudaDeviceSynchronize();

	

}

//-----------------------------------------------------------------------------------------------------------------------
__global__
void kernel_shift(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= IOutPutDim0 * IOutPutDim1 * IDim2)
	{
		return;
	}
	int iw = IOutPutDim1 * IDim2;
	int i_F = i / iw;
	int irest = i % iw;
	int i_dT = irest / IDim2;
	if (i_dT > d_iarr_deltaTLocal[i_F])
	{
		return;
	}
	int idx =  irest % IDim2;
	// claculation of bound index: 
	// arr_dT_ML[i_F, i_dT]
	// index of arr_dT_ML
	// arr_dT_ML is matrix with IOutPutDim0 rows and IOutPutDim1 cols
	int ind = i_F * IOutPutDim1 + i_dT;
	// !

	// calculation of:
	// d_Output[i_F][i_dT][idx] = d_input[2 * i_F][arr_dT_MI[i_F, i_dT]][idx]
	  // calculation num row of submatix No_2 * i_F of d_piarrInp = arr_dT_MI[ind]
	d_piarrOut[i] = d_piarrInp[2 * i_F * IDim1 * IDim2 +
	d_iarr_dT_MI[ind] * IDim2 + idx];
	
	if (idx >= d_iarr_dT_ML[ind])
	{
		int numRow = d_iarr_dT_RI[ind];
		int indInpMtrx = (2 * i_F + 1) * IDim1 * IDim2 + numRow * IDim2 + idx - d_iarr_dT_ML[ind];
		//atomicAdd(&d_piarrOut[i], d_piarrInp[ind]);
		d_piarrOut[i] += d_piarrInp[indInpMtrx];
	}
}
//def kernel_5_1(d_input, arr_deltaTLocal, arr_dT_MI, arr_dT_ML, arr_dT_RI, d_Output) :
//
//	i_F = cuda.blockIdx.x
//	i_dT = cuda.blockIdx.y
//	if i_dT > arr_deltaTLocal[i_F]:
//return
//
//idx = cuda.threadIdx.x
//
//
//if idx < arr_dT_ML[i_F, i_dT]:
//d_Output[i_F][i_dT][idx] = d_input[2 * i_F][arr_dT_MI[i_F, i_dT]][idx]
//else:
//d_Output[i_F][i_dT][idx] = d_input[2 * i_F][arr_dT_MI[i_F, i_dT]][idx] 
// + d_input[2 * i_F + 1][arr_dT_RI[i_F, i_dT]][idx - arr_dT_ML[i_F, i_dT]]
//--------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

void shift_and_sum(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut)
{
	int iarr_deltaTLocal[1000] = { 0 };
	cudaMemcpy(iarr_deltaTLocal, d_iarr_deltaTLocal, IOutPutDim0 * sizeof(int)
		, cudaMemcpyDeviceToHost);

	int* iarr_dT_ML = (int*)malloc(IOutPutDim0 * IOutPutDim1 * sizeof(int));
	cudaMemcpy(iarr_dT_ML, d_iarr_dT_ML, IOutPutDim0 * IOutPutDim1 * sizeof(int)
		, cudaMemcpyDeviceToHost);

	int* iarr_dT_MI = (int*)malloc(IOutPutDim0 * IOutPutDim1 * sizeof(int));
	cudaMemcpy(iarr_dT_MI, d_iarr_dT_MI, IOutPutDim0 * IOutPutDim1 * sizeof(int)
		, cudaMemcpyDeviceToHost);

	int* iarr_dT_RI = (int*)malloc(IOutPutDim0 * IOutPutDim1 * sizeof(int));
	cudaMemcpy(iarr_dT_RI, d_iarr_dT_RI, IOutPutDim0 * IOutPutDim1 * sizeof(int)
		, cudaMemcpyDeviceToHost);


	for (int i_F = 0; i_F < IOutPutDim0; ++i_F)
	{
		
		for (int i_dT = 0; i_dT < ( 1 +iarr_deltaTLocal[i_F]); ++i_dT)
		{
			int numRowOutputMtrxBegin0 = i_F * IOutPutDim1 * IDim2 + i_dT * IDim2;
			// number of element of beginning of the input 2 * i_F matrix's row with number 
			// dT_middle_index[i_F][i_dT]
			int numRowInputMtrxBegin0 = 2 * i_F * IDim1 * IDim2 + IDim2 * (iarr_dT_MI[i_F * IOutPutDim1 + i_dT]);
			cudaMemcpy(&d_piarrOut[numRowOutputMtrxBegin0], &d_piarrInp[numRowInputMtrxBegin0], IDim2 * sizeof(int)
				, cudaMemcpyDeviceToDevice);

			// number of beginning element of summated rows
			int numElemInRow = iarr_dT_ML[i_F * IOutPutDim1 + i_dT];
			// number of beginning element of output matrix  Output[i_F, i_dT, dT_middle_larger:]
			int numRowOutputMtrxBegin1 = numRowOutputMtrxBegin0 + numElemInRow;

			// number of the row of the submatrix of input matrix with number 2 * i_F + 1
			int numRowOfInputSubmatrix = iarr_dT_RI[i_F * IOutPutDim1 + i_dT];
			// number of beginning element of the input matrix Input[2 * i_F + 1, dT_rest_index, :i_T_max - dT_middle_larger]
			int numRowInputMtrxBegin1 = (2 * i_F + 1) * IDim1 * IDim2 + IDim2 * numRowOfInputSubmatrix ;
			int threadsPerBlock = 1024;
			int numberOfBlocks = (IDim2 - numElemInRow + threadsPerBlock - 1) / threadsPerBlock;
			sumArrays_ << <numberOfBlocks, threadsPerBlock >> > (&d_piarrOut[numRowOutputMtrxBegin1], &d_piarrInp[numRowInputMtrxBegin1], IDim2 - numElemInRow);
			cudaDeviceSynchronize();
		}
	}
	free(iarr_dT_RI);
	free(iarr_dT_ML);
	free(iarr_dT_MI);
}
//for i_F in range(F_jumps) :
//
//	f_start = (f_max - f_min) / float(F_jumps) * (i_F)+f_min
//	f_end = (f_max - f_min) / float(F_jumps) * (i_F + 1) + f_min
//	f_middle = (f_end - f_start) / 2. + f_start - correction
//	# it turned out in the end, that putting the correction + dF to f_middle_larger(or -dF / 2 to f_middle, and +dF / 2 to f_middle larger)
//	# is less sensitive than doing nothing when dedispersing a coherently dispersed pulse.
//	# The confusing part is that the hitting efficiency is better with the corrections(!? !).
//	f_middle_larger = (f_end - f_start) / 2 + f_start + correction
//	temp0 = (1. / f_start * *2 - 1. / (f_end) * *2)
//
//	val0 = -(1. / f_middle * *2 - 1. / f_start * *2) / temp0
//	val1 = -(1. / f_middle_larger * *2 - 1. / f_start * *2) / temp0
//	deltaTLocal = int(np.ceil((maxDT - 1) * temp0 / temp1))
//
//	for i_dT in range(deltaTLocal + 1) :
//		dT_middle_index = round(i_dT * val0)
//
//
//		dT_middle_larger = round(i_dT * val1)
//
//
//		dT_rest_index = i_dT - dT_middle_larger
//
//
//		i_T_min0 = 0
//
//		i_T_max0 = dT_middle_larger
//		#Output[i_F, i_dT + ShiftOutput, i_T_min0:i_T_max0] = Input[2 * i_F, dT_middle_index, i_T_min0:i_T_max0]
//		Output[i_F, i_dT, :dT_middle_larger] = Input[2 * i_F, dT_middle_index, :dT_middle_larger]
//
//		i_T_min = dT_middle_larger
//		i_T_max = T
//
//
//		#Output[i_F, i_dT + ShiftOutput, i_T_min:i_T_max] = Input[2 * i_F, dT_middle_index, i_T_min:i_T_max] + Input[2 * i_F + 1, dT_rest_index, i_T_min - dT_middle_larger:i_T_max - dT_middle_larger]
//		Output[i_F, i_dT, dT_middle_larger:] = Input[2 * i_F, dT_middle_index, dT_middle_larger:]
//      + Input[2 * i_F + 1, dT_rest_index, :i_T_max - dT_middle_larger]

__global__
void create_auxillary_1d_arrays(const int IFjumps, const int IMaxDT, const float VAlTemp1
	, const float VAlc2, const float VAlf_min, const float VAlcorrection
	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > IFjumps)
	{
		return;
	}
	float valf_start = VAlc2 * i + VAlf_min;
	float valf_end = valf_start + VAlc2;
	float valf_middle_larger = VAlc2 / 2. + valf_start + VAlcorrection;
	float valf_middle = VAlc2 / 2. + valf_start - VAlcorrection;
	float temp0 = 1. / (valf_start * valf_start) - 1. / (valf_end * valf_end);

	d_arr_val0[i] = -(1. / (valf_middle * valf_middle) - 1. / (valf_start * valf_start)) / temp0;

	d_arr_val1[i] = -(1. / (valf_middle_larger * valf_middle_larger)
		- 1. / (valf_start * valf_start)) / temp0;

	d_iarr_deltaTLocal[i] = (int)(ceil((((float)(IMaxDT)) - 1.) * temp0 / VAlTemp1));
}
//--------------------------------------------------------------------------------------
__global__
void kernel_2d_arrays(const int IDim0, const int IDim1
	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal
	, int* d_iarr_dT_middle_index, int* d_iarr_dT_middle_larger
	, int* d_iarr_dT_rest_index)

{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i > IDim0 * IDim1)
	{
		return;
	}
	int i_F = i / IDim1;
	int i_dT = i % IDim1;
	if (i_dT > (d_iarr_deltaTLocal[i_F]))
	{
		d_iarr_dT_middle_index[i] = 0;		
		d_iarr_dT_middle_larger[i] = 0;
		d_iarr_dT_rest_index[i] = 0;
		return;
	}

	d_iarr_dT_middle_index[i] = round(((float)i_dT) * d_arr_val0[i_F]);
	int ivalt = round(((float)i_dT) * d_arr_val1[i_F]);
	d_iarr_dT_middle_larger[i] = ivalt;
	d_iarr_dT_rest_index[i] = i_dT - ivalt;
	

}
//arr_dT_middle_index[cuda.blockIdx.x, cuda.threadIdx.x] = round(cuda.threadIdx.x * arr_val0[cuda.blockIdx.x])
//arr_dT_middle_larger[cuda.blockIdx.x, cuda.threadIdx.x] = round(cuda.threadIdx.x * arr_val1[cuda.blockIdx.x])
//arr_dT_rest_index[cuda.blockIdx.x, cuda.threadIdx.x] = cuda.threadIdx.x - arr_dT_middle_larger[cuda.blockIdx.x, cuda.threadIdx.x]
//--------------------------------------------------------------------------------------
void fnc_init(int* d_piarrImg, const int IImgrows, const int IImgcols
	, const int IDeltaT, int* d_piarrOut)
{
	// output in .npy:
		int* parr = (int*)malloc(IImgrows * IImgcols  * sizeof(int));
		cudaMemcpy(parr, d_piarrImg, IImgrows * IImgcols  * sizeof(int)
			, cudaMemcpyDeviceToHost);
		std::vector<int> v6(parr, parr + IImgrows * IImgcols );

		std::array<long unsigned, 1> leshape126 {IImgrows* IImgcols};

		npy::SaveArrayAsNumpy("init00.npy", false, leshape126.size(), leshape126.data(), v6);
		free(parr);

	cudaMemset(d_piarrOut, 0, IImgrows * IImgcols * (IDeltaT + 1) * sizeof(int));

	for (int i = 0; i < IImgrows; ++i)
	{
		{
			cudaMemcpy(&d_piarrOut[i * (IDeltaT + 1) * IImgcols], &d_piarrImg[i * IImgcols]
				, IImgcols * sizeof(int), cudaMemcpyDeviceToDevice);
		}
	}


	/*for (int i_dT = 1; i_dT < (IDeltaT + 1); ++i_dT)
		for (int iF = 0; iF < IImgrows; ++iF)
		{

			int threadsPerBlock = 1024;
			int numberOfBlocks = (i_dT + threadsPerBlock - 1) / threadsPerBlock;
			int* d_result = &d_piarrOut[iF * (IDeltaT + 1) * IImgcols + i_dT * IImgcols + i_dT];
			int* d_arg0 = &d_piarrOut[iF * (IDeltaT + 1) * IImgcols + (i_dT - 1) * IImgcols + i_dT];
			int* d_arg1 = &d_piarrImg[iF * IImgcols];
			sumArrays << <numberOfBlocks, threadsPerBlock >> > (d_result, d_arg0, d_arg1, IImgcols - i_dT);
			cudaDeviceSynchronize();
		}*/

}
//[F, T] = Image.shape
//
//deltaF = (f_max - f_min) / float(F)
//deltaT = int(np.ceil((maxDT - 1) * (1. / f_min * *2 - 1. / (f_min + deltaF) * *2) / (1. / f_min * *2 - 1. / f_max * *2)))
//
//Output = np.zeros([F, deltaT + 1, T], dtype = dataType)
//Output[:, 0, : ] = Image
//
//for i_dT in range(1, deltaT + 1) :
//	Output[:, i_dT, i_dT : ] = Output[:, i_dT - 1, i_dT : ] + Image[:, : -i_dT]
//-----------------------------------------------------------------------------
//CUDA kernel for element-wise summation
__global__ void sumArrays(int* d_result, const int* d_arr1, const int* d_arr2, int n)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
	{
		d_result[tid] = d_arr1[tid] + d_arr2[tid];
	}
}
//-----------------------------------------------------------------------------
//CUDA kernel for element-wise summation
__global__ void sumArrays_(int* d_result, const int* d_arr1, int n)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
	{
		d_result[tid] += d_arr1[tid] ;
	}
}

//-------------------------------------------------------------------------------------------------------------------------------
void fncCalcDimensionsOfOutputArrays(std::vector<int>* pivctOutDim0, std::vector<int>* pivctOutDim1
, std::vector<int>* pivctOutDim2,  const int IDim0, const int IDim1
, const int IDim2,  const int IMaxDT, const float VAlFmin
, const float VAlFmax)
{
	float val_dF = (VAlFmax - VAlFmin) / ((float)((*pivctOutDim0)[0]));
	
	
	
	for (int it = 1; it < pivctOutDim0->size(); ++it)
	{
		float valDeltaF = pow(2., it) * val_dF;
		float temp0 = 1. / (VAlFmin * VAlFmin) -
			1. / ((VAlFmin + valDeltaF) * (VAlFmin + valDeltaF));
		const float VAlTemp1 = 1. / (VAlFmin * VAlFmin) -
			1. / (VAlFmax * VAlFmax);
		int ideltaT = (int)(ceil(((float)IMaxDT - 1.0) * temp0 / VAlTemp1));
		(*pivctOutDim1)[it] = ideltaT + 1;
		(*pivctOutDim0)[it] = (*pivctOutDim0)[it-1] / 2;
		(*pivctOutDim2)[it] = (*pivctOutDim2)[it-1];

		
	}
}


