
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <chrono>


#define warpSize 32// CUDA warp size
const unsigned int   dataSize =  1 << 14;// Total number of elements
// Function to perform maximum reduction within a warp along with argmaximum
__device__ void warpMaxArgmax(int value, int idx, int* pmaxVal, int* pmaxIdx)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        int tempVal = __shfl_down_sync(0xffffffff, value, offset);
        int tempIdx = __shfl_down_sync(0xffffffff, idx, offset);
        if (tempVal > value)
        {
            value = tempVal;
            idx = tempIdx;
        }
    }
    if (threadIdx.x % warpSize == 0)
    {
        *pmaxVal = value;
        *pmaxIdx = idx;
    }
}

// Function to perform maximum reduction within a warp
__device__ int warpMax(int value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value = max(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

// Function to perform block-level reduction using shared memory
//template <unsigned int blockSize>
__global__ void blockMax(int* data, int* result) {
    extern __shared__ int sharedData[];

    int tid = threadIdx.x;
    int laneId = tid & (warpSize - 1); // Lane ID within the warp



    // Each warp performs a warp-level reduction
    int myValue = data[blockIdx.x * blockDim.x + threadIdx.x];
    int maxVal = warpMax(myValue);

    // Store warp-level maximum in shared memory
    if (laneId == 0) {
        sharedData[tid / warpSize] = maxVal;
    }
    __syncthreads();

    // Perform block-level reduction using shared memory
    if (tid < warpSize)
    {
        int blockMaxVal = (tid < (blockDim.x / warpSize)) ? sharedData[tid] : INT_MIN;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            blockMaxVal = max(blockMaxVal, __shfl_down_sync(0xffffffff, blockMaxVal, offset));
        }
        if (tid == 0)
        {
            result[blockIdx.x] = blockMaxVal;
            printf("blockIdx.x = %d ,max = %d\n", blockIdx.x, maxVal);
        }
    }
    __syncthreads();
}

// Function to perform block-level reduction using shared memory
//template <unsigned int blockSize>
__global__ void blockMaxArgMax(int* data, int* result,  int* argMax)
{
    extern __shared__ int sharedData[];

    int tid = threadIdx.x;
    int laneId = tid & (warpSize - 1); // Lane ID within the warp
    int warpId = tid / warpSize; // Warp ID


    // Each warp performs a warp-level reduction
    int myValue = data[blockIdx.x * blockDim.x + threadIdx.x];
    int myIdx = blockIdx.x * blockDim.x + tid;
    int maxVal, maxIdx;
    warpMaxArgmax(myValue, myIdx, &maxVal, &maxIdx);
    // int maxVal = warpMax(myValue);

    // Store warp-level maximum in shared memory
    if (laneId == 0) {
        sharedData[tid / warpSize] = maxVal;
        sharedData[blockDim.x / warpSize + warpId] = maxIdx;
    }
    __syncthreads();

    // Perform block-level reduction using shared memory
    if (tid < warpSize)
    {
        int blockMaxVal = (tid < (blockDim.x / warpSize)) ? sharedData[tid] : INT_MIN;
        int blockMaxIdx = (tid < (blockDim.x / warpSize)) ? sharedData[blockDim.x / warpSize + tid] : -1;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            //blockMaxVal = max(blockMaxVal, __shfl_down_sync(0xffffffff, blockMaxVal, offset));
            int tempVal = __shfl_down_sync(0xffffffff, blockMaxVal, offset);
            int tempIdx = __shfl_down_sync(0xffffffff, blockMaxIdx, offset);
            if (tempVal > blockMaxVal)
            {
                blockMaxVal = tempVal;
                blockMaxIdx = tempIdx;
            }
        }
        if (tid == 0)
        {
            result[blockIdx.x] = blockMaxVal;
            argMax[blockIdx.x] = blockMaxIdx;
            //printf("blockIdx.x = %d ,max = %d\n", blockIdx.x, blockMaxVal);
        }
    }
    __syncthreads();
}
//--------------------------------------------
__global__
void blockMaxArgMax_bu(int* d_arr 
    , int* d_pMaxArray, int* d_pAuxIndArray)
{
    extern __shared__ char buff[];
    int* arr_val = (int*)buff;
    int* iarr = (int*)(arr_val + blockDim.x);
    int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int nrow = blockIdx.y;
    for (int i = 0; i < blockDim.x; ++i)
    {
        arr_val[i] = d_arr[blockIdx.x * blockDim.x + i];
        iarr[i] = blockIdx.x * blockDim.x + i;
    }
    ////

    for (unsigned int s = (blockDim.x) / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (arr_val[tid] < arr_val[tid + s])
            {
                arr_val[tid] = arr_val[tid + s];
                iarr[tid] = iarr[tid + s];
            }
        }
        __syncthreads();
    }

    // Only thread 0 within each block computes the block's sum
    if (tid == 0)
    {
        d_pMaxArray[blockIdx.x] = arr_val[0];
        d_pAuxIndArray[blockIdx.x] = iarr[0];


    }      
    __syncthreads();
}
int main() 
{
    
    const int blockSize = 256; // Threads per block
    int* d_data =0;
    int* d_result =0; 
    int* d_argMax = 0;
    

    // Allocate memory on the device
    cudaMalloc(&d_data, dataSize * sizeof(int));
    cudaMalloc(&d_result, (dataSize + blockSize - 1) / blockSize * sizeof(int));
    cudaMalloc(&d_argMax, (dataSize + blockSize - 1) / blockSize * sizeof(int));

    int* data = (int*)malloc(dataSize * sizeof(int));
    int* result = (int*)malloc((dataSize + blockSize - 1) / blockSize * sizeof(int));
    int* argMax = (int*)malloc((dataSize + blockSize - 1) / blockSize * sizeof(int));
    for (int i = 0; i < dataSize; ++i)
    {
        data[i] = 1;
    }
    
    data[302] = 2; // Set the third element to 2
    cudaMemcpy(d_data, data, dataSize * sizeof(int), cudaMemcpyHostToDevice);

    

    // Launch the kernel with appropriate block and grid dimensions
    int gridSize = (dataSize + blockSize - 1) / blockSize;  

    int n_repeat = 50;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_repeat; ++i)
    {
        blockMaxArgMax << <gridSize, blockSize, blockSize / warpSize * sizeof(int) * 2 >> > (d_data, d_result, d_argMax);
        cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "blockMaxArgMax time:                     " << duration.count() / ((float)n_repeat) << " microseconds" << std::endl;

    cudaMemcpy(result, d_result, gridSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemcpy(argMax, d_argMax, gridSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_repeat; ++i)
    {
        blockMaxArgMax_bu << <gridSize, blockSize, blockSize * sizeof(int) * 2 >> > (d_data, d_result, d_argMax);
        cudaDeviceSynchronize();
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "blockMaxArgMax_bu time:                     " << duration.count() / ((float)n_repeat) << " microseconds" << std::endl;

    cudaMemcpy(result, d_result, gridSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(argMax, d_argMax, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Perform block-level reduction on the result array (optional, if multiple blocks)
    int gridSize1 = (gridSize + blockSize - 1) / blockSize;
    int* d_blockMax = 0;
    cudaMallocManaged(&d_blockMax, gridSize1 * sizeof(int));
    std::cout << "--------------------" << std::endl;
    // Perform block-level reduction kernel on the result array
    
   // blockMaxArgMax << <gridSize1, blockSize, blockSize / warpSize * sizeof(int)*2 >> > (d_result, d_blockMax, d_argMax);
   // cudaDeviceSynchronize();
    

    cudaFree(d_data);
    cudaFree(d_result);
    cudaFree(d_blockMax);
    cudaFree(d_argMax);
    free(data);
    free(result);
    free(argMax);
    return 0;
}
//
//#include <stdio.h>
//#include <cuda_runtime.h>
//
//const int warpSize = 32; // CUDA warp size
//
//// Function to perform maximum reduction within a warp along with argmaximum
//__device__ void warpMaxArgmax(int value, int idx, int& maxVal, int& maxIdx) {
//    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
//        int tempVal = __shfl_down_sync(0xffffffff, value, offset);
//        int tempIdx = __shfl_down_sync(0xffffffff, idx, offset);
//        if (tempVal > value) {
//            value = tempVal;
//            idx = tempIdx;
//        }
//    }
//    if (threadIdx.x % warpSize == 0) {
//        maxVal = value;
//        maxIdx = idx;
//    }
//}
//
//// Function to perform block-level reduction for maximum and argmaximum using shared memory
//template <unsigned int blockSize>
//__global__ void blockMaxArgmax(int* data, int* maxVal, int* maxIdx) {
//    extern __shared__ int sharedData[];
//
//    int tid = threadIdx.x;
//    int laneId = tid % warpSize; // Lane ID within the warp
//    int warpId = tid / warpSize; // Warp ID
//
//    int myValue = data[blockIdx.x * blockDim.x + tid];
//    int myIdx = blockIdx.x * blockDim.x + tid;
//
//    int warpMaxVal, warpMaxIdx;
//    warpMaxArgmax(myValue, myIdx, warpMaxVal, warpMaxIdx);
//
//    // Store warp-level maximum and argmaximum in shared memory
//    if (laneId == 0) {
//        sharedData[warpId] = warpMaxVal;
//        sharedData[blockDim.x / warpSize + warpId] = warpMaxIdx;
//    }
//    __syncthreads();
//
//    // Perform block-level reduction using shared memory for maximum and argmaximum
//    if (tid < warpSize) {
//        int blockMaxVal = (tid < (blockSize / warpSize)) ? sharedData[tid] : INT_MIN;
//        int blockMaxIdx = (tid < (blockSize / warpSize)) ? sharedData[blockDim.x / warpSize + tid] : -1;
//
//        warpMaxArgmax(blockMaxVal, blockMaxIdx, *maxVal, *maxIdx);
//    }
//}
//
//int main() {
//    const int blockSize = 256; // Threads per block
//    const int dataSize = 1024; // Total number of elements
//    int* d_data;
//    int* d_maxVal;
//    int* d_maxIdx;
//
//    // Allocate memory on the device
//    cudaMalloc(&d_data, dataSize * sizeof(int));
//    cudaMalloc(&d_maxVal, sizeof(int));
//    cudaMalloc(&d_maxIdx, sizeof(int));
//
//    // Initialize or copy data to the device
//    // ... (data initialization or copying omitted for brevity)
//
//    // Launch the kernel with appropriate block and grid dimensions
//    blockMaxArgmax<blockSize> << <(dataSize + blockSize - 1) / blockSize, blockSize, blockSize * sizeof(int) >> > (d_data, d_maxVal, d_maxIdx);
//
//    // Retrieve the result from device memory
//    int resultVal, resultIdx;
//    cudaMemcpy(&resultVal, d_maxVal, sizeof(int), cudaMemcpyDeviceToHost);
//    cudaMemcpy(&resultIdx, d_maxIdx, sizeof(int), cudaMemcpyDeviceToHost);
//
//    printf("Maximum value in block: %d at index: %d\n", resultVal, resultIdx);
//
//    cudaFree(d_data);
//    cudaFree(d_maxVal);
//    cudaFree(d_maxIdx);
//    return 0;
//}
