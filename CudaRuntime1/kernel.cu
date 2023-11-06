
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <math.h>


__global__ void addKernel(int n, float *c, const float *a, const float *b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    c[i] = a[i] + b[i];
}

    

    int main(void)
    {
        std::cout<<"HELLO1"<<std::endl;
        int N = 512; 
        float* x, * y, *z, *hz;

        // Allocate Unified Memory – accessible from CPU or GPU
        cudaMalloc(&x, N * sizeof(float));


      cudaMalloc(&y, N * sizeof(float));
        cudaMalloc(&z, N * sizeof(float));
        hz = (float*)malloc(N * sizeof(float));


        // initialize x and y arrays on the host
    float* hx, * hy;
    hx = (float*)malloc(N * sizeof(float));

    hy = (float*)malloc(N * sizeof(float));




       for (int i = 0; i < N; i++) {
            hx[i] = 1.0f;
            hy[i] = 2.0f;
            hz[i] = 10.;
        }

cudaMemcpy(x,hx, N * sizeof(float), cudaMemcpyHostToDevice);

cudaMemcpy(y,hy, N * sizeof(float), cudaMemcpyHostToDevice);

        // Run kernel on 1M elements on the GPU
       int threads = 1;
        int blocks = N;
        addKernel <<<blocks, threads >>>(N,z, x, y);

        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();
        cudaMemcpy(hz,z, N * sizeof(float), cudaMemcpyDeviceToHost);
        // Check for errors (all values should be 3.0f)
        float maxError = 0.0f;
       // for (int i = 0; i < N; i++)
            //maxError = fmax(maxError, fabs(hz[i] - 3.0f));
       // std::cout << "Max error: " << maxError << std::endl;


        for (int i = 0; i < N; ++i)
        {
            std::cout << hz[i]<<std::endl;
        }

        // Free memory
        free(hz);
        cudaFree(x);
        cudaFree(y);
        cudaFree(z);
       



        return 0;
    
}
