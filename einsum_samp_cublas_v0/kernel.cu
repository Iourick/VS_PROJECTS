
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
const int d = 3;
const int m = 4;
const int p = 2;
const int b = 378;
const int i = 100;
const int e = 10;
const int f = 100;

int main() {
    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Generate random complex-valued matrices on the GPU
    int size_dm = d * m * p * f;
    int size_impf = i * m * p * f;
    int size_dpe = d * p * e;
    int size_dfe = d * f * e;
    int size_d = d;

    cuComplex* w_dmpf = new cuComplex[size_dm];
    cuComplex* h_impf = new cuComplex[size_impf];
    cuComplex* F_dpe = new cuComplex[size_dpe];
    cuComplex* T_dfe = new cuComplex[size_dfe];
    cuComplex* D_d = new cuComplex[size_d];

    // Initialize matrices with random values
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < size_dm; ++i) {
        w_dmpf[i].x = static_cast<float>(rand()) / RAND_MAX;
        w_dmpf[i].y = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < size_impf; ++i) {
        h_impf[i].x = static_cast<float>(rand()) / RAND_MAX;
        h_impf[i].y = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < size_dpe; ++i) {
        F_dpe[i].x = static_cast<float>(rand()) / RAND_MAX;
        F_dpe[i].y = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < size_dfe; ++i) {
        T_dfe[i].x = static_cast<float>(rand()) / RAND_MAX;
        T_dfe[i].y = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < size_d; ++i) {
        D_d[i].x = static_cast<float>(rand()) / RAND_MAX;
        D_d[i].y = static_cast<float>(rand()) / RAND_MAX;
    }

    // Perform the equivalent operations using cuBLAS
    cuComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
    cuComplex beta = make_cuFloatComplex(0.0f, 0.0f);
    cuComplex* result = new cuComplex[size_impf * size_dpe * size_dfe];

    // Perform element-wise complex multiplication manually
    for (int idx = 0; idx < size_dm; ++idx) {
        if (idx < size_impf && idx < size_dpe && idx < size_d && idx < size_dfe) {
            result[idx].x = w_dmpf[idx].x * h_impf[idx].x * F_dpe[idx].x * D_d[idx % size_d].x
                - w_dmpf[idx].y * h_impf[idx].y * F_dpe[idx].x * D_d[idx % size_d].y;
            result[idx].y = w_dmpf[idx].x * h_impf[idx].y * F_dpe[idx].x * D_d[idx % size_d].x
                + w_dmpf[idx].y * h_impf[idx].x * F_dpe[idx].x * D_d[idx % size_d].y;
        }
        else {
            result[idx].x = 0.0f; // Set to 0 for elements that are out of bounds
            result[idx].y = 0.0f;
        }
    }

    // Perform matrix multiplication with T_dfe
    cublasCgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_C, i, e, d, &alpha,
        result, i, size_impf * size_dpe, T_dfe, i, size_dfe, &beta, result, i, size_impf * size_dpe, p * f);

    // Clean up
    delete[] w_dmpf;
    delete[] h_impf;
    delete[] F_dpe;
    delete[] T_dfe;
    delete[] D_d;
    delete[] result;

    cublasDestroy(handle);

    return 0;
}


