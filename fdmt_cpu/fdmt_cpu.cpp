// fdmt_cpu.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <iostream>


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
#include <chrono>
#include <stdlib.h>
#include "fileInput.h"
#include "FdmtFuncs.h"

using namespace std;



int main()
{
	char strFolder[] = "";
	char strOutImageNpyFile[] = "out_image.npy";

	int* piarr = (int*)malloc(sizeof(int));
	
	int** ppiarrImage = &piarr;
	
	*ppiarrImage = (int*)realloc(*ppiarrImage, 40 * sizeof(int));
	
	int iMaxDT = 0;
	int iImRows = 0, iImCols = 0;
	float val_fmax = 0., val_fmin = 0.;
	int ireturn = downloadInputData(strFolder, strOutImageNpyFile, &iMaxDT, ppiarrImage, &iImRows, &iImCols,
		&val_fmin, &val_fmax);
	switch (ireturn)
	{
	case 1:
		cout << "Err. Can't allocate memory for input image " << std::endl;
		return 1;
	case 2:
		cout << "Err. Input dimensions must be a power of 2 " << std::endl;
		return 1;
	case 0:
		cout << "Input data downloaded properly " << std::endl;
		break;
	default:
		break;
	}
	const int IMaxDT = iMaxDT;
	const int IImgrows = iImRows;
	const int IImgcols = iImCols;
	const float VAlFmin = val_fmin;
	const float VAlFmax = val_fmax;



	int* piarrImage = *ppiarrImage;
	// 6. allocate memory for device array

	int* piarrImOut = (int*)malloc(IImgcols * IMaxDT * sizeof(int)); 
	// !6

	// 7. calculations	
	clock_t start = clock();	
	

	fncFdmt_cu_v0(piarrImage, IImgrows, IImgcols
		, VAlFmin, VAlFmax, IMaxDT, piarrImOut);
	/*clock_t end = clock();
	double duration = double(end - start) / CLOCKS_PER_SEC;
	std::cout << "Time taken by function fncFdmt_cu_v0: " << duration << " seconds" << std::endl;*/
	
	// !7
	// output in .npy:

	std::vector<int> v1(piarrImOut, piarrImOut + IImgcols * IMaxDT);

	std::array<long unsigned, 2> leshape101 {IImgcols,IMaxDT};

	npy::SaveArrayAsNumpy("out_image.npy", false, leshape101.size(), leshape101.data(), v1);
	free(piarrImOut);
	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
