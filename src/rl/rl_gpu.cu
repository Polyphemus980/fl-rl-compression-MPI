#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "rl_gpu.cuh"
#include "../utils.cuh"

namespace RunLength
{
    // Main functions

    RLCompressed gpuCompress(uint8_t *data, size_t size)
    {
        if (size == 0)
        {
            return RLCompressed{
                .outputValues = nullptr,
                .outputCounts = nullptr,
                .count = 0};
        }

        // Copy input data to GPU
        uint8_t *d_data;
        CHECK_CUDA(cudaMalloc(&d_data, sizeof(uint8_t) * size));
        CHECK_CUDA(cudaMemcpy(d_data, data, sizeof(uint8_t) * size, cudaMemcpyHostToDevice));

        // Prepare GPU arrays
        uint32_t *d_startMask;
        CHECK_CUDA(cudaMalloc(&d_startMask, sizeof(uint32_t) * size));
        CHECK_CUDA(cudaMemset(d_startMask, 0, sizeof(uint32_t) * size));
        uint32_t *d_scannedStartMask;
        CHECK_CUDA(cudaMalloc(&d_scannedStartMask, sizeof(uint32_t) * size));
        uint32_t *d_startIndices;
        CHECK_CUDA(cudaMalloc(&d_startIndices, sizeof(uint32_t) * size));
        uint32_t *d_startIndicesLength;
        CHECK_CUDA(cudaMalloc(&d_startIndicesLength, sizeof(uint32_t)));
        // We could do it only after we know how much exactly we need, but it doesn't really matter
        // as we will copy back exact amount back to cpu anyway.
        // This way error handling is easier as all allocations are done at the beggining of the function.
        uint8_t *d_outputValues;
        CHECK_CUDA(cudaMalloc(&d_outputValues, sizeof(uint8_t) * size));
        uint8_t *d_outputCounts;
        CHECK_CUDA(cudaMalloc(&d_outputCounts, sizeof(uint8_t) * size));
        // Same here, we could wait and allocate it later with exact size, but this way it's easier
        // to handle errors.
        uint32_t *d_recalculateSequence;
        CHECK_CUDA(cudaMalloc(&d_recalculateSequence, sizeof(uint32_t) * size));
        uint32_t *d_shouldRecalculate;
        CHECK_CUDA(cudaMalloc(&d_shouldRecalculate, sizeof(uint32_t)));
        CHECK_CUDA(cudaMemset(d_shouldRecalculate, 0, sizeof(uint32_t)));

        // Calculate start mask
        const uint32_t calculateStartMaskThreadsCount = 1024;
        const uint32_t calculateStartMaskBlocksCount = ceil(size * 1.0 / calculateStartMaskThreadsCount);
        compressCalculateStartMask<<<calculateStartMaskBlocksCount, calculateStartMaskThreadsCount>>>(d_data, size, d_startMask);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());

        // Calculate scanned start mask
        compressCalculateScannedStartMask(d_startMask, d_scannedStartMask, size);

        // Calculate start indicies
        const uint32_t calculateStartIndiciesThreadsCount = 1024;
        const uint32_t calculateStartIndiciesBlocksCount = ceil(size * 1.0 / calculateStartIndiciesThreadsCount);
        compressCalculateStartIndicies<<<calculateStartIndiciesBlocksCount, calculateStartIndiciesThreadsCount>>>(d_scannedStartMask, size, d_startIndices, d_startIndicesLength);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());

        // First copy to CPU size of final output to know how much bytes to copy (and allocate)
        // and to know how big kernel should be
        uint32_t outputSize = 0;
        CHECK_CUDA(cudaMemcpy(&outputSize, d_startIndicesLength, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // Check if we need to recalculate some sequence due to size > 255
        const uint32_t checkForMoreSequencesThreadsCount = 1024;
        const uint32_t checkForMoreSequencesBlocksCount = ceil(outputSize * 1.0 / checkForMoreSequencesThreadsCount);
        compressCheckForMoreSequences<<<checkForMoreSequencesBlocksCount, checkForMoreSequencesThreadsCount>>>(d_startIndices, d_startIndicesLength, d_recalculateSequence, d_shouldRecalculate);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());

        // Copy to cpu boolean value to check if need to recalculate some sequences
        uint32_t shouldRecalculate = 0;
        CHECK_CUDA(cudaMemcpy(&shouldRecalculate, d_shouldRecalculate, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        if (shouldRecalculate != 0)
        {
            // TODO: recalculate sequences
        }

        // Calculate final output
        const uint32_t calculateOutputThreadsCount = 1024;
        const uint32_t calculateOutputBlocksCount = ceil(outputSize * 1.0 / calculateOutputThreadsCount);
        compressCalculateOutput<<<calculateOutputBlocksCount, calculateOutputThreadsCount>>>(d_data, size, d_startIndices, d_startIndicesLength, d_outputValues, d_outputCounts);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());

        // Allocate needed cpu arrays
        uint8_t *outputValues = reinterpret_cast<uint8_t *>(malloc(sizeof(uint8_t) * outputSize));
        if (outputValues == nullptr)
        {
            throw std::runtime_error("Cannot allocate memory");
        }
        uint8_t *outputCounts = reinterpret_cast<uint8_t *>(malloc(sizeof(uint8_t) * outputSize));
        if (outputCounts == nullptr)
        {
            throw std::runtime_error("Cannot allocate memory");
        }

        // Copy results to CPU
        CHECK_CUDA(cudaMemcpy(outputValues, d_outputValues, sizeof(uint8_t) * outputSize, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(outputCounts, d_outputCounts, sizeof(uint8_t) * outputSize, cudaMemcpyDeviceToHost));

        // Deallocate GPU arrays
        cudaFree(d_data);
        cudaFree(d_startMask);
        cudaFree(d_scannedStartMask);
        cudaFree(d_startIndices);
        cudaFree(d_startIndicesLength);
        cudaFree(d_outputValues);
        cudaFree(d_outputCounts);

        return RLCompressed{
            .outputValues = outputValues,
            .outputCounts = outputCounts,
            .count = outputSize,
        };
    }

    RLDecompressed gpuDecompress(uint8_t *values, uint8_t *counts, size_t size)
    {
        // TODO:
    }

    // Kernels

    __global__ void compressCalculateStartMask(uint8_t *d_data, size_t size, uint32_t *d_startMask)
    {
        auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
        if (threadId == 0 || (threadId > 0 && threadId < size && d_data[threadId] != d_data[threadId - 1]))
        {
            d_startMask[threadId] = 1;
        }
    }

    __global__ void compressCalculateStartIndicies(uint32_t *d_scannedStartMask, size_t size, uint32_t *d_startIndicies, uint32_t *d_startIndiciesLength)
    {
        __shared__ uint32_t s_maxLength[1];
        auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
        auto localThreadId = threadIdx.x;

        // Initialize shared memory
        if (localThreadId == 0)
        {
            // It will always be at least 1, in case of length 0 we early return from main compress function
            s_maxLength[0] = 1;
        }
        __syncthreads();

        if (threadId == 0)
        {
            d_startIndicies[0] = 0;
        }
        else if (threadId < size && d_scannedStartMask[threadId] != d_scannedStartMask[threadId - 1])
        {
            auto id = d_scannedStartMask[threadId] - 1;
            d_startIndicies[id] = threadId;
            // + 1 because we want the length, not the index
            atomicMax(&s_maxLength[0], id + 1);
        }
        __syncthreads();

        // Save currently biggest changed index in global variable
        if (localThreadId == 0)
        {
            atomicMax(d_startIndiciesLength, s_maxLength[0]);
        }
    }

    __global__ void compressCheckForMoreSequences(uint32_t *d_startIndicies, uint32_t *d_startIndiciesLength, uint32_t *d_recalculateSequence, uint32_t *d_shouldRecalculate)
    {
        __shared__ uint32_t s_shouldRecalculate[1];
        __shared__ uint32_t s_startIndiciesLength[1];
        auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
        auto localThreadId = threadIdx.x;

        // Initialize shared memory
        if (localThreadId == 0)
        {
            s_shouldRecalculate[0] = false;
            s_startIndiciesLength[0] = d_startIndiciesLength[0];
        }
        __syncthreads();

        if (threadId < s_startIndiciesLength[0] - 1)
        {
            auto diff = d_startIndicies[threadId + 1] - d_startIndicies[threadId];
            if (diff > 255)
            {
                d_recalculateSequence[threadId] = diff / 255;
                atomicOr(s_shouldRecalculate, 1);
            }
        }
        __syncthreads();

        // Save result from shared to global memory
        if (localThreadId == 0)
        {
            atomicOr(d_shouldRecalculate, s_shouldRecalculate[0]);
        }
    }

    __global__ void compressCalculateOutput(uint8_t *d_data, size_t size, uint32_t *d_startIndicies, uint32_t *d_startIndiciesLength, uint8_t *d_outputValues, uint8_t *d_outputCounts)
    {
        __shared__ uint32_t s_length[1];
        auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
        auto localThreadId = threadIdx.x;

        // Initialize shared memory
        if (localThreadId == 0)
        {
            s_length[0] = d_startIndiciesLength[0];
        }
        __syncthreads();

        if (threadId < s_length[0])
        {
            d_outputValues[threadId] = d_data[d_startIndicies[threadId]];
        }

        if (threadId == s_length[0] - 1)
        {
            d_outputCounts[threadId] = (uint8_t)((uint32_t)size - d_startIndicies[threadId]);
        }
        else if (threadId < s_length[0] - 1)
        {
            d_outputCounts[threadId] = d_startIndicies[threadId + 1] - d_startIndicies[threadId];
        }
    }

    // Helpers

    void compressCalculateScannedStartMask(uint32_t *d_startMask, uint32_t *d_scannedStartMask, size_t size)
    {
        thrust::inclusive_scan(thrust::device, d_startMask, d_startMask + size, d_scannedStartMask);
    }

} // RunLength