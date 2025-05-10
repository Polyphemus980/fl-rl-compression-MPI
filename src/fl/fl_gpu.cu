#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <stdexcept>
#include <mpi.h>

#include "fl_gpu.cuh"
#include "fl_common.cuh"
#include "../utils.cuh"
#include "../timers/cpu_timer.cuh"
#include "../timers/gpu_timer.cuh"

namespace FixedLength
{
    // Main functions
    FLCompressed gpuCompress(uint8_t *data, size_t size)
    {
        if (size == 0)
        {
            return FLCompressed();
        }

        std::exception error;
        bool isError = false;

        Timers::CpuTimer cpuTimer;
        Timers::GpuTimer gpuTimer;

        size_t bitsSize = ceil(size * 1.0 / FRAME_LENGTH);
        size_t valuesSize = 0;

        // GPU arrays
        uint8_t *d_data = nullptr;
        uint8_t *d_outputBits = nullptr;
        uint64_t *d_frameStartIndiciesBits = nullptr;
        uint8_t *d_outputValues = nullptr;

        // CPU arrays
        uint8_t *outputBits = nullptr;
        uint8_t *outputValues = nullptr;

        try
        {
            gpuTimer.start();

            // Allocate arrays on GPU
            CHECK_CUDA(cudaMalloc(&d_data, sizeof(uint8_t) * size));
            CHECK_CUDA(cudaMalloc(&d_outputBits, sizeof(uint8_t) * bitsSize));
            CHECK_CUDA(cudaMalloc(&d_frameStartIndiciesBits, sizeof(uint64_t) * bitsSize));

            gpuTimer.end();
            gpuTimer.printResult("Allocate arrays on GPU");

            gpuTimer.start();

            // Copy input to GPU
            CHECK_CUDA(cudaMemcpy(d_data, data, sizeof(uint8_t) * size, cudaMemcpyHostToDevice));

            gpuTimer.end();
            gpuTimer.printResult("Copy input data to GPU");

            gpuTimer.start();

            // Calculate outputBits
            constexpr size_t outputBitsThreadsPerBlock = BLOCK_SIZE;
            const size_t outputBitsBlocksCount = ceil(size * 1.0 / outputBitsThreadsPerBlock);
            compressCalculateOutputBits<<<outputBitsBlocksCount, outputBitsThreadsPerBlock>>>(d_data, size, d_outputBits, bitsSize);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaGetLastError());

            // Calculate frameStartIndiciesBits
            constexpr size_t frameStartIndiciesThreadsPerBlock = BLOCK_SIZE;
            const size_t frameStartIndiciesBlocksCount = ceil(bitsSize * 1.0 / frameStartIndiciesThreadsPerBlock);
            compressInitializeFrameStartIndiciesBits<<<frameStartIndiciesBlocksCount, frameStartIndiciesThreadsPerBlock>>>(d_frameStartIndiciesBits, d_outputBits, bitsSize);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaGetLastError());
            compressCalculateFrameStartIndiciesBits(d_frameStartIndiciesBits, bitsSize);

            // Calculate length of outputValues array
            uint8_t outputBitsLast = 0;
            CHECK_CUDA(cudaMemcpy(&outputBitsLast, &d_outputBits[bitsSize - 1], sizeof(uint8_t), cudaMemcpyDeviceToHost));
            uint64_t frameStartIndiciesBitsLast = 0;
            CHECK_CUDA(cudaMemcpy(&frameStartIndiciesBitsLast, &d_frameStartIndiciesBits[bitsSize - 1], sizeof(uint64_t), cudaMemcpyDeviceToHost));
            uint64_t lastFrameElementCount = size % FRAME_LENGTH == 0 ? FRAME_LENGTH : (size - (size / FRAME_LENGTH) * FRAME_LENGTH);
            valuesSize = ceil((frameStartIndiciesBitsLast + lastFrameElementCount * outputBitsLast) * 1.0 / 8);

            // Allocate gpu array for `outputValues`
            CHECK_CUDA(cudaMalloc(&d_outputValues, sizeof(uint8_t) * valuesSize));
            CHECK_CUDA(cudaMemset(d_outputValues, 0, sizeof(uint8_t) * valuesSize));

            constexpr size_t outputValuesThreadsPerBlock = BLOCK_SIZE;
            const size_t outputValuesBlocksCount = ceil(size * 1.0 / outputValuesThreadsPerBlock);
            compressCalculateOutput<<<outputValuesBlocksCount, outputValuesThreadsPerBlock>>>(d_data, size, d_outputBits, bitsSize, d_frameStartIndiciesBits, d_outputValues, valuesSize);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaGetLastError());

            gpuTimer.end();
            gpuTimer.printResult("Compression");

            cpuTimer.start();

            // Allocate arrays on CPU
            outputBits = reinterpret_cast<uint8_t *>(malloc(sizeof(uint8_t) * bitsSize));
            if (outputBits == nullptr)
            {
                throw std::runtime_error("Cannot allocate memory");
            }
            outputValues = reinterpret_cast<uint8_t *>(malloc(sizeof(uint8_t) * valuesSize));
            if (outputValues == nullptr)
            {
                throw std::runtime_error("Cannot allocate memory");
            }

            cpuTimer.end();
            cpuTimer.printResult("Allocate arrays on CPU");

            gpuTimer.start();

            // Copy results to CPU
            CHECK_CUDA(cudaMemcpy(outputBits, d_outputBits, sizeof(uint8_t) * bitsSize, cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(outputValues, d_outputValues, sizeof(uint8_t) * valuesSize, cudaMemcpyDeviceToHost));

            gpuTimer.end();
            gpuTimer.printResult("Copy results to CPU");
        }
        catch (const std::exception &e)
        {
            error = e;
            isError = true;
        }

        gpuTimer.start();

        // Deallocate gpu arrays
        cudaFree(d_data);
        cudaFree(d_outputBits);
        cudaFree(d_frameStartIndiciesBits);
        cudaFree(d_outputValues);

        gpuTimer.end();
        gpuTimer.printResult("Deallocate ararys on GPU");

        if (isError)
        {
            throw error;
        }

        return FLCompressed(outputBits, bitsSize, outputValues, valuesSize, size);
    }

    FLDecompressed gpuDecompress(size_t outputSize, uint8_t *bits, size_t bitsSize, uint8_t *values, size_t valuesSize)
    {
        if (valuesSize == 0 || bitsSize == 0 || outputSize == 0)
        {
            return FLDecompressed();
        }

        Timers::CpuTimer cpuTimer;
        Timers::GpuTimer gpuTimer;

        std::exception error;
        bool isError = false;

        // CPU arrays
        uint8_t *data;

        // GPU arrays
        uint8_t *d_bits;
        uint8_t *d_values;
        uint64_t *d_frameStartIndiciesBits;
        uint8_t *d_data;

        try
        {
            cpuTimer.start();

            // Allocate array on CPU
            data = reinterpret_cast<uint8_t *>(malloc(sizeof(uint8_t) * outputSize));
            if (data == nullptr)
            {
                throw std::runtime_error("Cannot allocate memory");
            }

            cpuTimer.end();
            cpuTimer.printResult("Allocate arrays on CPU");

            gpuTimer.start();

            // Allocate arrays on GPU
            CHECK_CUDA(cudaMalloc(&d_bits, sizeof(uint8_t) * bitsSize));
            CHECK_CUDA(cudaMalloc(&d_values, sizeof(uint8_t) * valuesSize));
            CHECK_CUDA(cudaMalloc(&d_frameStartIndiciesBits, sizeof(uint64_t) * bitsSize));
            CHECK_CUDA(cudaMalloc(&d_data, sizeof(uint8_t) * outputSize));

            gpuTimer.end();
            gpuTimer.printResult("Allocate arrays on GPU");

            gpuTimer.start();

            // Copy input to GPU
            CHECK_CUDA(cudaMemcpy(d_bits, bits, sizeof(uint8_t) * bitsSize, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_values, values, sizeof(uint8_t) * valuesSize, cudaMemcpyHostToDevice));

            gpuTimer.end();
            gpuTimer.printResult("Copy input to GPU");

            gpuTimer.start();

            // Calculate frameStartIndiciesBits
            constexpr size_t frameStartIndiciesThreadsPerBlock = BLOCK_SIZE;
            const size_t frameStartIndiciesBlocksCount = ceil(bitsSize * 1.0 / frameStartIndiciesThreadsPerBlock);
            compressInitializeFrameStartIndiciesBits<<<frameStartIndiciesBlocksCount, frameStartIndiciesThreadsPerBlock>>>(d_frameStartIndiciesBits, d_bits, bitsSize);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaGetLastError());
            compressCalculateFrameStartIndiciesBits(d_frameStartIndiciesBits, bitsSize);

            // Calculate output
            constexpr size_t outputThreadsPerBlock = BLOCK_SIZE;
            const size_t outputBlocksCount = ceil(outputSize * 1.0 / outputThreadsPerBlock);
            decompressCalculateOutput<<<outputBlocksCount, outputThreadsPerBlock>>>(d_data, outputSize, d_bits, bitsSize, d_values, valuesSize, d_frameStartIndiciesBits);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaGetLastError());

            gpuTimer.end();
            gpuTimer.printResult("Decompression");

            gpuTimer.start();

            // Copy result to CPU
            CHECK_CUDA(cudaMemcpy(data, d_data, sizeof(uint8_t) * outputSize, cudaMemcpyDeviceToHost));

            gpuTimer.end();
            gpuTimer.printResult("Copy results to CPU");
        }
        catch (const std::exception &e)
        {
            error = e;
            isError = true;
        }

        gpuTimer.start();

        // Deallocate GPU arrays
        cudaFree(d_bits);
        cudaFree(d_values);
        cudaFree(d_frameStartIndiciesBits);
        cudaFree(d_data);

        gpuTimer.end();
        gpuTimer.printResult("Deallocate arrays on GPU");

        if (isError)
        {
            throw error;
        }

        return FLDecompressed(data, outputSize);
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

        // Push results back to global memoryd_frameStartIndiciesBits
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

    __global__ void compressCalculateOutput(uint8_t *d_data, size_t size, uint8_t *d_outputBits, size_t bitsSize, uint64_t *d_frameStartIndiciesBits, uint8_t *d_outputValues, size_t valuesSize)
    {
        auto threadId = blockDim.x * blockIdx.x + threadIdx.x;

        // Don't follow if threadId is outside of data scope
        if (threadId >= size)
        {
            return;
        }

        // Encode data
        uint64_t frameId = threadId / FRAME_LENGTH;
        uint64_t frameElementId = threadId % FRAME_LENGTH;
        uint8_t requiredBits = d_outputBits[frameId];
        uint64_t bitsOffset = d_frameStartIndiciesBits[frameId] + frameElementId * (uint64_t)requiredBits;
        size_t outputId = bitsOffset / 8;
        uint8_t outputOffset = bitsOffset % 8;
        uint8_t value = d_data[threadId];
        uint8_t encodedValue = (value << outputOffset);
        atomicOrUint8t(&d_outputValues[outputId], encodedValue);
        // If it overflows encode the overflowed part on next byte
        if (outputOffset + requiredBits > 8)
        {
            uint8_t overflowValue = (value >> (8 - outputOffset));
            atomicOrUint8t(&d_outputValues[outputId + 1], overflowValue);
        }
    }

    __global__ void decompressCalculateOutput(uint8_t *d_data, size_t size, uint8_t *d_bits, size_t bitsSize, uint8_t *d_values, size_t valuesSize, uint64_t *d_frameStartIndiciesBits)
    {
        auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
        // Don't follow if threadId is outside of data scope
        if (threadId >= size)
        {
            return;
        }

        // Decode data
        uint64_t frameId = threadId / FRAME_LENGTH;
        uint64_t frameElementId = threadId % FRAME_LENGTH;
        uint8_t usedBits = d_bits[frameId];
        uint64_t bitsOffset = d_frameStartIndiciesBits[frameId] + frameElementId * usedBits;
        size_t inputId = bitsOffset / 8;
        uint8_t inputOffset = bitsOffset % 8;
        uint8_t mask = (1 << usedBits) - 1;
        uint8_t decodedValue = (d_values[inputId] >> inputOffset) & mask;
        // If it overflow decode the overflowed part of the next byte
        if (inputOffset + usedBits > 8)
        {
            uint8_t overflowBits = inputOffset + usedBits - 8;
            uint8_t overflowMask = (1 << overflowBits) - 1;
            uint8_t overflowValue = (d_values[inputId + 1] & overflowMask) << (usedBits - overflowBits);
            decodedValue |= overflowValue;
        }
        d_data[threadId] = decodedValue;
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

    __device__ uint8_t atomicOrUint8t(uint8_t *address, uint8_t val)
    {
        unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
        unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
        unsigned int sel = selectors[(size_t)address & 3];
        unsigned int old, assumed, new_, current_val, updated_val;
        old = *base_address;
        do
        {
            assumed = old;
            current_val = (uint8_t)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440);
            updated_val = current_val | val;
            new_ = __byte_perm(old, updated_val, sel);

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