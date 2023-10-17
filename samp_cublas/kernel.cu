
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
/*To link your program with cuBLAS in Visual Studio 2022 on Windows, you need to perform the following steps:

Install CUDA Toolkit:

Before you can use cuBLAS, make sure you have installed the CUDA Toolkit on your system. You can download it from the official NVIDIA website: https://developer.nvidia.com/cuda-downloads

Create a CUDA Project:

If you don't already have a CUDA project, create a new CUDA project in Visual Studio.

a. Open Visual Studio 2022.

b. Go to "File" > "New" > "Project..."

c. Select "CUDA" under "Installed" > "Visual C++" > "NVIDIA."

d. Choose a CUDA Runtime version compatible with your GPU, and configure other project settings as needed.

Include cuBLAS Header:

In your CUDA source file (usually with a .cu extension), include the cuBLAS header at the beginning:

cpp
Copy code
#include <cublas_v2.h>
Link with cuBLAS Library:

You need to configure your project to link with the cuBLAS library.

a. Right-click on your CUDA project in Solution Explorer and select "Properties."

b. In the Project Properties window, go to "Configuration Properties" > "VC++ Directories."

c. Under "Library Directories," add the path to the cuBLAS library directory. It typically looks something like this (adjust the version number according to your CUDA installation):

javascript
Copy code
$(CUDA_PATH_V11_0)\lib\x64
d. Next, go to "Configuration Properties" > "Linker" > "Input."

e. Under "Additional Dependencies," add the cuBLAS library file. You may need to specify both the static and dynamic versions, depending on your project requirements. For example:

cublas.lib
f. Click "Apply" or "OK" to save the project properties.

Build and Run:
*/

// Function to initialize a matrix with random values
void initializeMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Function to print a matrix
void printMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}

int main() {
    const int N = 3; // Matrix size (N x N)

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Create input matrices A and B
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * N]; // Result matrix

    // Initialize A and B with random values
    initializeMatrix(A, N, N);
    initializeMatrix(B, N, N);

    // Print input matrices A and B
    std::cout << "Matrix A:" << std::endl;
    printMatrix(A, N, N);
    std::cout << "Matrix B:" << std::endl;
    printMatrix(B, N, N);

    // Perform matrix multiplication using cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);

    // Print the result matrix C
    std::cout << "Result Matrix C:" << std::endl;
    printMatrix(C, N, N);

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;
    cublasDestroy(handle);

    return 0;
}

