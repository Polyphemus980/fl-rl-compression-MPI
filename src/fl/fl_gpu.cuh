#ifndef FL_GPU_H
#define FL_GPU_H

#include "fl_common.cuh"

namespace FixedLength
{
    // consts
    static constexpr size_t BLOCK_SIZE = 1024;
    ;

    // Main functions
    FLCompressed gpuCompress(uint8_t *data, size_t size);
    FLDecompressed gpuDecompress(size_t outputSize, uint8_t *bits, size_t bitsSize, uint8_t *values, size_t valuesSize);

    // Kernels
    __global__ void compressCalculateOutputBits(uint8_t *d_data, size_t size, uint8_t *d_outputBits, size_t bitsSize);
    __global__ void compressInitializeFrameStartIndiciesBits(uint64_t *d_frameStartIndiciesBits, uint8_t *d_outputBits, size_t bitsSize);
    __global__ void compressCalculateOutput(uint8_t *d_data, size_t size, uint8_t *d_outputBits, size_t bitsSize, uint64_t *d_frameStartIndiciesBits, uint8_t *outputValues, size_t valuesSize);

    // Helpers
    __device__ uint8_t atomicMaxUint8t(uint8_t *address, uint8_t val);
    __device__ uint8_t atomicOrUint8t(uint8_t *address, uint8_t val);
    void compressCalculateFrameStartIndiciesBits(uint64_t *d_frameStartIndiciesBits, size_t bitsSize);
}

#endif // FL_GPU_H