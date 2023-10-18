
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <math.h>
#include <stdio.h>


#include <stdlib.h> // Supports dynamic memory management.

#include <array>
#include <iostream>
#include <string>
#include <vector>

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
	bool fortran_order = false;
	// 1. loading typeofdata
	std::vector<unsigned long> shape {};
	std::vector<int> ivctDataType_maxDT;
	npy::LoadArrayFromNumpy("iarrDataType_maxDT.npy", shape, fortran_order, ivctDataType_maxDT);
	const int IMaxDT = ivctDataType_maxDT[1];
	// !1

	// 2. loading XX
	std::vector<unsigned long> shape1 {};
	std::vector<int> vctXX;
	npy::LoadArrayFromNumpy("XX.npy", shape1, fortran_order, vctXX);
	// !2

	// 3. loading shape
	std::vector<unsigned long> shape0 {};
	std::vector<int> ivctImShape;
	npy::LoadArrayFromNumpy("iarrShape.npy", shape0, fortran_order, ivctImShape);
	// !3



	// 4. loading fmin and maxDT  fmin_maxDT.npy
	std::vector<unsigned long> shape2 {};
	std::vector<float> vctfmin_max;
	npy::LoadArrayFromNumpy("fmin_max.npy", shape2, fortran_order, vctfmin_max);
	// ! 4

	// 5. creating dynamic array arrXX
	int* piarrImage = new int[vctXX.size()];
	for (int i = 0; i < vctXX.size(); ++i)
	{
		piarrImage[i] = vctXX[i];
		//piarrImage[i] = vctXX[i / ivctImShape[1]][i % ivctImShape[1]];
	}
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
    std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;
	// !7


	


	
//-------------------------------------------------------------------------------
//------  otput image -------------------------------------------------------------------------
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
	IROWS = ivctImShape[1];
	ICOLS = ivctDataType_maxDT[1];
	// zaglushka
	memcpy(piarrImOut, piarrImage, IROWS * ICOLS * sizeof(int));
	// !zaglushka
	

	int imax = *std::max_element(piarrImage, piarrImage + ivctImShape[0] * ivctImShape[1]);
	float coeff = 255. / (double(imax));
    ivctOut = std::vector<std::vector<int>>(IROWS, std::vector<int>(ICOLS, 0)); // Initialize with your data
    for (int i = 0; i < IROWS; ++i)
    for (int  j = 0; j < ICOLS; ++j)
    {
        ivctOut[i][j] = (int)(coeff * piarrImOut[i * ICOLS + j]);
        
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
void fncFdmt_cu_v0(int* piarrImgInp, const int iImgrows, const int iImgcols
	, const float f_min, const  float f_max, const int imaxDT, int* piarrImgOut)
{
	float deltaF = (f_max - f_min) / ((float)(iImgrows));
	int ideltaT = int(ceil((imaxDT - 1.) * (1. / (f_min * f_max)
		- 1. / ((f_min + deltaF)
			* (f_min + deltaF))) / (1. / (f_min
				* f_min) - 1. / (f_max * f_max))));
	// 1. declare pointers to device arrays
	int* d_p0 = 0;
	int* d_p1 = 0;
	int* d_piarrOut_0 = 0;
	int* d_piarrOut_1 = 0;
	int* d_piarrImgInp = 0;
	// !1

		
	// 2. allocate memory to device arrays
	clock_t start = clock();
	cudaMalloc(&d_piarrOut_0, iImgrows * (ideltaT + 1) * iImgcols * sizeof(int));
	cudaMalloc(&d_piarrOut_1, iImgrows * (ideltaT + 1) * iImgcols * sizeof(int));
	cudaMalloc(&d_piarrImgInp, iImgrows  * iImgcols * sizeof(int));
	// !2
	clock_t end = clock();
	double duration = double(end - start) / CLOCKS_PER_SEC;
	std::cout << "Time taken by cudaMalloc: " << duration << " seconds" << std::endl;


	// 3  Initialize the device arrays with zeros
	start = clock();
	cudaMemset(d_piarrOut_0, 0, iImgrows * (ideltaT + 1) * iImgcols * sizeof(int));
	cudaMemset(d_piarrOut_1, 0, iImgrows * (ideltaT + 1) * iImgcols * sizeof(int));
	// !3
	end = clock();
	duration = double(end - start) / CLOCKS_PER_SEC;
	std::cout << "Time taken by cudaMemset: " << duration << " seconds" << std::endl;

	// 4.copy input data from host to device
	start = clock();
	cudaMemcpy(d_piarrImgInp, piarrImgInp, iImgrows * iImgcols * sizeof(int)
		, cudaMemcpyHostToDevice);
	end = clock();
	duration = double(end - start) / CLOCKS_PER_SEC;
	std::cout << "Time taken by cudaMemcpy: " << duration << " seconds" << std::endl;
	// !4

	// 5. call initialization func
	fnc_init(d_piarrImgInp, f_min, f_max, imaxDT, iImgrows , iImgcols, ideltaT + 1, d_piarrOut_0);
	//! 5
	
	// 6. quant iteration's calculation
	const int I_F = (int)(log2((double)(iImgrows)));
	// !6

	// 7.pointers initialization
	d_p0 = d_piarrOut_0;
	d_p1 = d_piarrOut_1;
	// 7!

	// 8. allocate memory to device  auxiliary arrays
	start = clock();
	float* d_arr_val0 = 0;
	cudaMalloc(&d_arr_val0, iImgrows / 2 * sizeof(float));

	float* d_arr_val1 = 0;
	cudaMalloc(&d_arr_val1, iImgrows / 2 * sizeof(float));

	int* d_arr_deltaTLocal = 0;
	cudaMalloc(&d_arr_deltaTLocal, iImgrows / 2 * sizeof(int));

	int* d_arr_dT_MI = 0;
	cudaMalloc(&d_arr_dT_MI, iImgrows / 2 * (ideltaT + 1) * sizeof(int));

	int* d_arr_dT_ML = 0;
	cudaMallocManaged(&d_arr_dT_ML, iImgrows / 2 * (ideltaT + 1) * sizeof(int));

	int* d_arr_dT_RI = 0;
	cudaMalloc(&d_arr_dT_RI, iImgrows / 2 * (ideltaT + 1) * sizeof(int));
	end = clock();
	duration = double(end - start) / CLOCKS_PER_SEC;
	std::cout << "Time taken by allocate memory: " << duration << " seconds" << std::endl;
	// !8

	for (int iit = 1; iit < (I_F + 1); ++iit)
	{
		/*fncFdmtIteration(d_p0, IDim0, const int IDim1
			, const int IDim2, const int IMaxDT, const float VAlFmin
			, const float VAlFmax, const int ITerNum, float* d_arr_val0
			, float* d_arr_val1, int* d_iarr_deltaTLocal
			, int* d_iarr_dT_MI, int* d_iarr_dT_ML, int* d_iarr_dT_RI
			, d_p1);*/

		// exchange order of pointers
		int* d_pt = d_p0;
		d_p0 = d_p1;
		d_p1 = d_pt;
		// !

	}

	start = clock();
	cudaFree(d_arr_val0);
	cudaFree(d_arr_val1);
	cudaFree(d_arr_deltaTLocal);
	cudaFree(d_arr_dT_MI);
	cudaFree(d_arr_dT_ML);
	cudaFree(d_arr_dT_RI);

	
	cudaFree(d_piarrOut_0);
	cudaFree(d_piarrOut_1);
	cudaFree(d_piarrImgInp);

	end = clock();
	duration = double(end - start) / CLOCKS_PER_SEC;
	std::cout << "Time taken by cudaFree: " << duration << " seconds" << std::endl;

}
//--------------------------------------------------------------------------------------
__global__
void create_auxillary_1d_arrays(const int IFjumps, const int IMaxDT, const float VAlTemp1
	, const float VAlc2, const float VAlf_min, const float VAlcorrection
	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= IFjumps)
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
void create_auxillary_2d_arrays(const int IDim0, const int IDim1
	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal
	,int* d_iarr_dT_middle_index, int* d_iarr_dT_middle_larger
	, int* d_arr_dT_rest_index)
	
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= IDim0 * IDim1)
	{
		return;
	}
	int i_F = i / IDim1;
	int i_dT = i % IDim1;
	if(i_dT > d_iarr_deltaTLocal[i_F])
	{
		return;
	}
		
	d_iarr_dT_middle_index[i] = round(i_dT * d_arr_val0[i_F]);
	int ivalt = round(i_dT * d_arr_val1[i_F]);
	d_iarr_dT_middle_larger[i] = ivalt;
	d_arr_dT_rest_index[i] = i_dT - ivalt;

}
//--------------------------------------------------------------------------------------

void fnc_init(int* d_piarrImg, const float f_min, const  float f_max
	, const int imaxDT, const int IImgrows, const int IImgcols
	, const int IDimOut1,int* d_piarrOut)
{
	cudaMemset(d_piarrOut, 0, IImgrows * IImgcols *IDimOut1 *sizeof(int));
	//Output[:, 0, : ] = Image
	//        for i_dT in range(1, deltaT + 1) :
	//            Output[:, i_dT, i_dT : ] = Output[:, i_dT - 1, i_dT : ] + Image[:, : -i_dT]
	//            return Output

	
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
void fncFdmtIteration(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, const int IMaxDT, const float VAlFmin
	, const float VAlFmax, const int ITerNum, float* d_arr_val0
	, float* d_arr_val1, int* d_iarr_deltaTLocal
	, int* d_iarr_dT_MI,  int* d_iarr_dT_ML,  int* d_iarr_dT_RI
	, int* d_piarrOut)
{
	float val_dF = (VAlFmax - VAlFmin) / ((float)(IDim0));
	float valDeltaF = pow(2., ITerNum) * val_dF;
	float temp0 = 1. / (VAlFmin * VAlFmin) -
		1. / ((VAlFmin + valDeltaF) * (VAlFmin + valDeltaF));
	const float VAlTemp1 = 1. / (VAlFmin * VAlFmin) -
		1. / (VAlFmax * VAlFmax);
	int ideltaT = (int)(ceil(((float)IMaxDT - 1.0) * temp0 / VAlTemp1));
	int iOutPutDim1 = ideltaT + 1;
	int iOutPutDim0 = IDim0 / 2;
	int iOutPutDim2 = IDim2;
	

	// set zeros in output array
	cudaMemset(d_piarrOut, 0, iOutPutDim0 * iOutPutDim1 * iOutPutDim2 * sizeof(int));
	// !
	 
	float val_correction = 0;
	if (ITerNum > 0)
	{
		val_correction = val_dF / 2.;
	}

	int iF_jumps = iOutPutDim0;
	// 9. auxiliary constants initialization
	const float VAlC2 = (VAlFmax - VAlFmin) / ((float)(iF_jumps));
	// !9

	// 10. calculating first 3 auxillary 1 dim arrays
	int threadsPerBlock = 256;
	int numberOfBlocks = (iF_jumps + threadsPerBlock - 1) / threadsPerBlock;
	create_auxillary_1d_arrays << <numberOfBlocks, threadsPerBlock >> > (iF_jumps
		, IMaxDT, VAlTemp1, VAlC2, VAlFmin, val_correction
		, d_arr_val0, d_arr_val1, d_iarr_deltaTLocal);
	// !10

	// 11. calculating second 3 auxillary 2 dim arrays
	int quantEl = iF_jumps * iOutPutDim1;
	 threadsPerBlock = 256;
	numberOfBlocks = (quantEl + threadsPerBlock - 1) / threadsPerBlock;
	
	create_auxillary_2d_arrays<<<numberOfBlocks, threadsPerBlock >> > (iF_jumps
		,iOutPutDim1,d_arr_val0, d_arr_val1,d_iarr_deltaTLocal
		,d_iarr_dT_MI,  d_iarr_dT_ML, d_iarr_dT_RI);
	// !11

    // 12. 
	quantEl = IDim0 * IDim1 * IDim2;
	numberOfBlocks = (quantEl + threadsPerBlock - 1) / threadsPerBlock;
	kernel_shift << <numberOfBlocks, threadsPerBlock >> > (d_piarrInp
		, IDim0, IDim1, IDim2, d_iarr_deltaTLocal, d_iarr_dT_MI, d_iarr_dT_ML, d_iarr_dT_RI
		, d_piarrOut);

}
__global__
void kernel_shift(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2,int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI
	, int* d_piarrOut)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= IDim0 * IDim1 * IDim2)
	{
		return;
	}
	int iw = IDim1 * IDim2;
	int i_F = i / iw;
	int irest = i % iw;
	int i_dT = irest / IDim2;
	if (i_dT > d_iarr_deltaTLocal[i_F])
	{
		return;
	}
	int ic =  irest % IDim2;

	d_piarrOut[i] = d_piarrInp[2 * i_F * iw +
		d_iarr_dT_MI[i_F * IDim1 + i_dT] * IDim1 + ic];
	// index of i_F and i_dT
	int i_FT = i_F * IDim1 + i_dT;
	if (ic >= d_iarr_dT_ML[i_FT])
	{
		int ind = (2 * i_F + 1) * iw + d_iarr_dT_RI[i_FT] * IDim2 + ic - d_iarr_dT_ML[i_FT];
		d_piarrOut[i] = d_piarrInp[ind];
	}

}


