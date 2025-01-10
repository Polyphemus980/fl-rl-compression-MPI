#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scan.h>

#include "fl_gpu.cuh"
#include "../utils.cuh"

namespace FixedLength
{
    // Main functions
    FLCompressed gpuCompress(uint8_t *data, size_t size)
    {
        if (size == 0)
        {
            return FLCompressed{
                .outputBits = nullptr,
                .bitsSize = 0,
                .outputValues = nullptr,
                .valuesSize = 0,
                .inputSize = 0};
        }

        // Copy input to GPU
        uint8_t *d_data;
        CHECK_CUDA(cudaMalloc(&d_data, sizeof(uint8_t) * size));
        CHECK_CUDA(cudaMemcpy(d_data, data, sizeof(uint8_t) * size, cudaMemcpyHostToDevice));

        // Allocate arrays on GPU
        size_t bitsSize = ceil(size * 1.0 / FRAME_LENGTH);
        uint8_t *d_outputBits;
        CHECK_CUDA(cudaMalloc(&d_outputBits, sizeof(uint8_t) * bitsSize));
        uint64_t *d_frameStartIndiciesBits;
        CHECK_CUDA(cudaMalloc(&d_frameStartIndiciesBits, sizeof(uint64_t) * bitsSize));

        // Calculate outputBits
        constexpr size_t outputBitsThreadsPerBlock = 1024;
        const size_t outputBitsBlocksCount = ceil(size * 1.0 / outputBitsThreadsPerBlock);
        compressCalculateOutputBits<<<outputBitsBlocksCount, outputBitsThreadsPerBlock>>>(d_data, size, d_outputBits, bitsSize);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());

        // FIXME: remove me, only for testing
        {
            uint8_t *outputBitsCPU = reinterpret_cast<uint8_t *>(malloc(sizeof(uint8_t) * bitsSize));
            CHECK_CUDA(cudaMemcpy(outputBitsCPU, d_outputBits, sizeof(uint8_t) * bitsSize, cudaMemcpyDeviceToHost));
            printf("bits size: %lu\n", bitsSize);
            printf("output Bits: \n");
            for (size_t i = 0; i < bitsSize; i++)
            {
                printf("%hhu\n", outputBitsCPU[i]);
            }
        }

        // Calculate frameStartIndiciesBits
        constexpr size_t frameStartIndiciesThreadsPerBlock = 1024;
        const size_t frameStartIndiciesBlocksCount = ceil(bitsSize * 1.0 / frameStartIndiciesThreadsPerBlock);
        compressInitializeFrameStartIndiciesBits<<<frameStartIndiciesBlocksCount, frameStartIndiciesThreadsPerBlock>>>(d_frameStartIndiciesBits, d_outputBits, bitsSize);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());
        compressCalculateFrameStartIndiciesBits(d_frameStartIndiciesBits, bitsSize);

        // FIXME: remove me, only for testing
        {
            uint64_t *frameStartIndiciesBitsCPU = reinterpret_cast<uint64_t *>(malloc(sizeof(uint64_t) * bitsSize));
            CHECK_CUDA(cudaMemcpy(frameStartIndiciesBitsCPU, d_frameStartIndiciesBits, sizeof(uint64_t) * bitsSize, cudaMemcpyDeviceToHost));
            printf("frameStartIndiciesBits: \n");
            for (size_t i = 0; i < bitsSize; i++)
            {
                printf("%lu\n", frameStartIndiciesBitsCPU[i]);
            }
        }

        // TODO: finish

        // Deallocate gpu arrays
        cudaFree(d_data);
        cudaFree(d_outputBits);
        cudaFree(d_frameStartIndiciesBits);

        // TODO: fill it
        return FLCompressed{
            .outputBits = nullptr,
            .bitsSize = 0,
            .outputValues = nullptr,
            .valuesSize = 0,
            .inputSize = 0};
    }

    // Kernels
    __global__ void compressCalculateOutputBits(uint8_t *d_data, size_t size, uint8_t *d_outputBits, size_t bitsSize)
    {
        auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
        auto localThreadId = threadIdx.x;

        // Don't follow if threadId is outside of data scope
        if (threadId >= size)
        {
            return;
        }

        constexpr size_t BLOCK_SIZE = 1024;
        constexpr size_t FRAMES_PER_BLOCK = BLOCK_SIZE / FRAME_LENGTH;

        auto frameId = threadId / FRAME_LENGTH;
        auto localFrameId = frameId - blockIdx.x * FRAMES_PER_BLOCK;

        __shared__ uint8_t s_outputBits[FRAMES_PER_BLOCK];

        // Initialize shared memory
        // We always need at least 1
        if (localThreadId < FRAMES_PER_BLOCK)
        {
            s_outputBits[localThreadId] = 1;
        }
        __syncthreads();

        // Calculate number of required bits
        uint8_t requiredBits = 8 - countLeadingZeroes(d_data[threadId]);
        atomicMaxUint8t(&s_outputBits[localFrameId], requiredBits);
        __syncthreads();

        // Push results back to global memory
        auto globalId = blockIdx.x * FRAMES_PER_BLOCK + localThreadId;
        if (localThreadId < FRAMES_PER_BLOCK && globalId < bitsSize)
        {
            atomicMaxUint8t(&d_outputBits[globalId], s_outputBits[localThreadId]);
        }
    }

    __global__ void compressInitializeFrameStartIndiciesBits(uint64_t *d_frameStartIndiciesBits, uint8_t *d_outputBits, size_t bitsSize)
    {
        auto threadId = blockDim.x * blockIdx.x + threadIdx.x;

        // Don't follow if threadId is outside of data scope
        if (threadId >= bitsSize)
        {
            return;
        }

        d_frameStartIndiciesBits[threadId] = d_outputBits[threadId] * FRAME_LENGTH;
    }

    // Helpers
    __device__ uint8_t atomicMaxUint8t(uint8_t *address, uint8_t val)
    {
        unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
        unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
        unsigned int sel = selectors[(size_t)address & 3];
        unsigned int old, assumed, max_, new_;
        old = *base_address;
        do
        {
            assumed = old;
            max_ = max(val, (uint8_t)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440));
            new_ = __byte_perm(old, max_, sel);

            if (new_ == old)
                break;

            old = atomicCAS(base_address, assumed, new_);

        } while (assumed != old);

        return old;
    }

    void compressCalculateFrameStartIndiciesBits(uint64_t *d_frameStartIndiciesBits, size_t bitsSize)
    {
        thrust::exclusive_scan(thrust::device, d_frameStartIndiciesBits, d_frameStartIndiciesBits + bitsSize, d_frameStartIndiciesBits);
    }

}