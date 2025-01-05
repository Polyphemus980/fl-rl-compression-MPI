#ifndef RL_GPU_H
#define RL_GPU_H

#include "rl_common.cuh"

namespace RunLength
{
    // Main functions
    RLCompressed gpuCompress(uint8_t *data, size_t size);
    RLDecompressed gpuDecompress(uint8_t *values, uint8_t *counts, size_t size);

    // Kernels
    __global__ void compressCalculateStartMask(uint8_t *d_data, size_t size, size_t *d_startMask);

    // Helpers

} // RunLength

#endif // RL_GPU_H