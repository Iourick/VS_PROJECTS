
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cufft.h>
#include <complex>
#include <vector>

void printComplexData(cufftComplex* data, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << "(" << data[i].x << ", " << data[i].y << ") ";
    }
    std::cout << std::endl;
}

int main() {
    const int nrows = 5; // Number of rows
    const int ncols = 8; // Number of columns in the matrix
    const int size = ncols; // Assuming each row has ncols elements

    cufftComplex* matrixData = new cufftComplex[nrows * ncols]; // Input data
    cufftComplex* outputData = new cufftComplex[nrows * ncols]; // Output data

    // Initialize cuFFT
    cufftHandle plan;
    int n[1] = { size };
    cufftResult result = cufftPlanMany(&plan, 1, n, NULL, 1, ncols, NULL, 1, ncols, CUFFT_C2C, nrows);
    if (result != CUFFT_SUCCESS) {
        std::cerr << "Error creating cuFFT plan\n";
        delete[] matrixData;
        delete[] outputData;
        return -1;
    }

    // Generate random input data
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < nrows * ncols; ++i) {
        matrixData[i].x = static_cast<float>(rand()) / RAND_MAX;
        matrixData[i].y = static_cast<float>(rand()) / RAND_MAX;
    }

    std::cout << "Input Data:\n";
    for (int i = 0; i < nrows; ++i) {
        std::cout << "Row " << i << ": ";
        printComplexData(matrixData + i * ncols, ncols);
    }

    // Execute forward FFT
    result = cufftExecC2C(plan, matrixData, outputData, CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS) {
        std::cerr << "Error performing forward FFT\n";
        cufftDestroy(plan);
        delete[] matrixData;
        delete[] outputData;
        return -1;
    }

    std::cout << "\nForward FFT Output:\n";
    for (int i = 0; i < nrows; ++i) {
        std::cout << "Row " << i << ": ";
        printComplexData(outputData + i * ncols, ncols);
    }

    // Execute inverse FFT
    result = cufftExecC2C(plan, outputData, matrixData, CUFFT_INVERSE);
    if (result != CUFFT_SUCCESS) {
        std::cerr << "Error performing inverse FFT\n";
        cufftDestroy(plan);
        delete[] matrixData;
        delete[] outputData;
        return -1;
    }

    std::cout << "\nInverse FFT Output:\n";
    for (int i = 0; i < nrows; ++i) {
        std::cout << "Row " << i << ": ";
        printComplexData(matrixData + i * ncols, ncols);
    }

    // Clean up resources
    cufftDestroy(plan);
    delete[] matrixData;
    delete[] outputData;
    ////////////
//#define ARRAY_SIZE 8
//    cufftHandle plan1;
//    cufftComplex* arr0, * arrTransf, * arr1;
//
//    // Allocate memory for the arrays (assuming complex data)
//    
//    
//    arr1 = (cufftComplex*)malloc(sizeof(cufftComplex) * ARRAY_SIZE);
//
//    using scalar_type = float;
//    using data_type = std::complex<scalar_type>;
//
//    std::vector<data_type> data(8, 0);
//    std::vector<data_type> data1(8, 0);
//
//    for (int i = 0; i < 8; i++) {
//        data[i] = data_type(i, -i);
//    }
//
//    std::printf("Input array:\n");
//    for (auto& i : data) {
//        std::printf("%f + %fj\n", i.real(), i.imag());
//    }
//    std::printf("=====\n");
//
//    cufftComplex* d_data = nullptr;
//    
//    cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(data_type) * data.size());
//    cudaMalloc(reinterpret_cast<void**>(&arrTransf), sizeof(data_type) * data.size());
//    cudaMalloc(reinterpret_cast<void**>(&arr1), sizeof(data_type) * data.size());
//    cudaMemcpyAsync(d_data, data.data(), sizeof(data_type) * data.size(),
//        cudaMemcpyHostToDevice);
//
//    // Initialize arr0 with some data (for example)
//    // ... (initialize arr0 with data)
//
//    // Create the CUFFT plan for 1D complex-to-complex FFT
//    cufftPlan1d(&plan1, ARRAY_SIZE, CUFFT_C2C, 1);
//
//    // Perform forward FFT on arr0 to get arrTransf
//    cufftExecC2C(plan1, d_data, arrTransf, CUFFT_FORWARD);
//    // Copy data from device (cufftComplex) to host (std::complex<float>)
//    
//
//    
//    cudaMemcpyAsync(data.data(), arrTransf, 8 * sizeof(cufftComplex),
//        cudaMemcpyDeviceToHost);
//    // Perform inverse FFT on arrTransf to get arr1
//    cufftExecC2C(plan1, arrTransf, arr1, CUFFT_INVERSE);
//
//    // Destroy the plan
//    cufftDestroy(plan1);
//
//    cudaMemcpyAsync(data1.data(), arr1, sizeof(data_type) * data.size(),
//        cudaMemcpyDeviceToHost );
//
//    
//
//    // Free allocated memory
//    free(arr0);
//    free(arrTransf);
//    free(arr1);


    return 0;
}

