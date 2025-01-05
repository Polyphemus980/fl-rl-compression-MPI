#include "rl_gpu.cuh"
#include "../utils.cuh"

namespace RunLength
{
    RLCompressed gpuCompress(uint8_t *data, size_t size)
    {
        // Copy input data to GPU
        uint8_t *d_data;
        CHECK_CUDA(cudaMalloc(&d_data, sizeof(uint8_t) * size));
        CHECK_CUDA(cudaMemcpy(d_data, data, sizeof(uint8_t) * size, cudaMemcpyHostToDevice));

        // Prepare GPU arrays
        size_t *d_startMask;
        CHECK_CUDA(cudaMalloc(&d_startMask, sizeof(size_t) * size));
        CHECK_CUDA(cudaMemset(d_startMask, 0, sizeof(size_t) * size));
        size_t *d_scannedStartMask;
        CHECK_CUDA(cudaMalloc(&d_scannedStartMask, sizeof(size_t) * size));
        // We allocate `size` elements so that we could do it all in one kernel
        // the actual size of `startIndices` in most cases will be much smaller.
        size_t *d_startIndices;
        CHECK_CUDA(cudaMalloc(&d_startIndices, sizeof(size_t) * size));
        size_t *d_startIndicesLength;
        CHECK_CUDA(cudaMalloc(&d_startIndicesLength, sizeof(size_t)));

        // Calculate start mask
        const uint32_t calculateStartMaskThreadsCount = 1024;
        const uint32_t calculateStartMaskBlocksCount = ceil(size * 1.0 / calculateStartMaskThreadsCount);
        compressCalculateStartMask<<<calculateStartMaskBlocksCount, calculateStartMaskThreadsCount>>>(d_data, size, d_startMask);
        CHECK_CUDA(cudaGetLastError());

        // TODO: rest

        // Wait for GPU to finish calculations
        CHECK_CUDA(cudaDeviceSynchronize());

        // TODO: copy results from GPU to CPU

        // Deallocate GPU arrays
        cudaFree(d_data);
        cudaFree(d_startMask);
        cudaFree(d_scannedStartMask);
        cudaFree(d_startIndices);
        cudaFree(d_startIndicesLength);
    }

    RLDecompressed gpuDecompress(uint8_t *values, uint8_t *counts, size_t size)
    {
        // TODO:
    }

    __global__ void compressCalculateStartMask(uint8_t *d_data, size_t size, size_t *d_startMask)
    {
        auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
        if (threadId == 0 || threadId > 0 && d_data[threadId] != d_data[threadId - 1])
        {
            d_startMask[threadId] = 1;
        }
    }

} // RunLength