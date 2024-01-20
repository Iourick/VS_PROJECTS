
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <random>


#define TILE_DIM 16
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

//--------------------------------------------------------------------------------------
void MatrTransp(float* parrA, int nRows, int nCols, float* parrRez)
{
    float* pA = parrA;
    for (int i = 0; i < nRows; i++)
    {

        float* pRezt = parrRez + i;
        for (int j = 0; j < nCols; j++)
        {
            *pRezt = *pA;
            pA++;
            pRezt += nRows;
        }

    }
}

//---------------------------------------------------
__global__ void transpose(float* input, float* output, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // Shared memory tile

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Transpose data from global to shared memory
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] =  input[y * width + x];
    }
    __syncthreads();

    // Calculate new indices for writing to output
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Transpose data from shared to global memory
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

//----------------------------------------------------

__global__
void calcMultiTransposition_kernel(float* output, const int height, const int width, float* input) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // Shared memory tile

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int ichan = blockIdx.z;
    // Transpose data from global to shared memory
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[ichan * height * width + y * width + x];
    }
    __syncthreads();

    // Calculate new indices for writing to output
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Transpose data from shared to global memory
    if (x < height && y < width) {
        output[ichan * height * width + y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}
//----------------------------------------------------

__global__
void calcPowerMtrx_kernel(float* output, const int height, const int width,  const int npol, float* input) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // Shared memory tile

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int ichan = blockIdx.z;
    // Transpose data from global to shared memory
    if (x < width && y < height)
    {
        float sum = 0.;
        for (int i = 0; i < npol; ++i)
        {
            sum += input[(ichan * npol +i)*height * width + y * width + x];
        }

        tile[threadIdx.y][threadIdx.x] = sum;
    }
    __syncthreads();

    // Calculate new indices for writing to output
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Transpose data from shared to global memory
    if (x < height && y < width) {
        output[ichan * height * width + y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}
//------------------------------------------

__global__
void calcPartSum_kernel(float* d_parr_out, const int lenChunk, const int npol_physical, float* d_parr_inp)
{
    
    int ichan = blockIdx.y;
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < lenChunk)
    {
        float sum = 0;
        for (int i = 0; i < npol_physical; ++i)
        {
            sum += d_parr_inp[(ichan * npol_physical + i) * lenChunk + ind];
        }
        d_parr_out[ichan * lenChunk + ind] =  sum;
    }
}

int main()
{
    /*int nchan = 64;
    int npol = 2;
    int len_chunk = 1<<10; 
    int n_p = 16;*/
    int nchan = 64;
    int npol = 2;// 2;
    int len_chunk =  1 << 18;
    int n_p = 4; 

    int lenarr = nchan * npol * len_chunk;
    float* arr = (float*)malloc(lenarr * sizeof(float));
    // Random number generation setup
    //std::random_device rd;
    //std::mt19937 gen(rd());
    //std::uniform_real_distribution<float> dis(0.0f, 1.0f); // Range between 0 and 1

    //// Fill the float array with random values
    //for (int i = 0; i < lenarr; ++i) {
    //    arr[i] = dis(gen);
    //}
    for (int i = 0; i < lenarr; ++i)
    {
        arr[i] = (float)i + 1.;
    }

    float* d_arr = 0;
    float* d_out = 0;

    int lenout = nchan * len_chunk;
    float* out = (float*)malloc(nchan * len_chunk * sizeof(float));

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaMalloc((void**)&d_arr, lenarr * sizeof(float));
    cudaMalloc((void**)&d_out, lenout * sizeof(float));
    cudaMemcpy(d_arr, arr, lenarr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(&d_out,0, lenout * sizeof(float));

    //nchan = 1;
    /*dim3 threadsPerBlock(TILE_DIM, TILE_DIM, 1);
    dim3 blocksPerGrid((n_p + TILE_DIM - 1) / TILE_DIM, (len_chunk/ n_p + TILE_DIM - 1) / TILE_DIM, nchan);
    size_t sz = TILE_DIM * (TILE_DIM + 1) * npol *sizeof(float);
    calcPowerMtrx_kernel__<<< blocksPerGrid, threadsPerBlock,sz>>>(d_out, len_chunk, npol
        , n_p, d_arr);*/
    
    dim3 threadsPerBlock0(1024, 1);
    dim3 blocksPerGrid0((len_chunk + threadsPerBlock0.x - 1) / threadsPerBlock0.x,  nchan);
    calcPartSum_kernel << < blocksPerGrid0, threadsPerBlock0 >> > (d_out, len_chunk, npol,  d_arr);
    cudaDeviceSynchronize();

    float* d_out1 = 0;
    cudaMalloc((void**)&d_out1, lenout * sizeof(float));
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM, 1);
    dim3 blocksPerGrid((n_p + TILE_DIM - 1) / TILE_DIM, (len_chunk / n_p + TILE_DIM - 1) / TILE_DIM, nchan);
    size_t sz = TILE_DIM * (TILE_DIM + 1) * sizeof(float);
    calcMultiTransposition_kernel << < blocksPerGrid, threadsPerBlock,sz >> > (d_out1, len_chunk / n_p, n_p,  d_out);



    cudaMemcpy(out, d_out1, lenout * sizeof(float), cudaMemcpyDeviceToHost);

    calcPowerMtrx_kernel << < blocksPerGrid, threadsPerBlock, sz >> > (d_out, len_chunk / n_p, n_p,npol, d_arr);
    cudaMemcpy(out, d_out, lenout * sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(d_arr);
    cudaFree(d_out);
    cudaFree(d_out1);
    

    float* out1 = (float*)malloc(lenout * sizeof(float));
    float* out2 = (float*)malloc(lenout * sizeof(float));
    float* out3 = (float*)malloc(lenout * sizeof(float));
    for (int i = 0; i < nchan; ++i)
    {        
        for (int j = 0; j < len_chunk; ++j)
        {
            out1[i * len_chunk + j] = 0.;
            for (int k = 0; k < npol; ++k)
            {
                out1[i * len_chunk + j] += arr[(i * npol + k) * len_chunk + j];
            }
        }
    }

    for (int i = 0; i < nchan; ++i)
    {
        MatrTransp(&out1[i * len_chunk], len_chunk / n_p, n_p, &out2[i * len_chunk]);
    }

    for (int i = 0; i < lenout; ++i)
    {
        out3[i] = fabs(out2[i] - out[i]);
    }

    float max = -1.;
    for (int i = 0; i < lenout; ++i)
    {
        if (out3[i] > max)
        {
            max = out3[i];
        }
    }




    free(out);
    free(arr);
    free(out1);
    free(out2);
    free(out3);


    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
