#ifndef RL_GPU_H
#define RL_GPU_H

#include "rl_common.cuh"

namespace RunLength
{
    // Main functions
    RLCompressed gpuCompress(uint8_t *data, size_t size);
    RLDecompressed gpuDecompress(uint8_t *values, uint8_t *counts, size_t size);

    // Kernels
    __global__ void compressCalculateStartMask(uint8_t *d_data, size_t size, uint32_t *d_startMask);
    __global__ void compressCalculateStartIndicies(uint32_t *d_scannedStartMask, size_t size, uint32_t *d_startIndicies, uint32_t *d_startIndiciesLength);
    __global__ void compressCheckForMoreSequences(uint32_t *d_startIndicies, uint32_t *d_startIndiciesLength, uint32_t *d_recalculateSequence, uint32_t *d_shouldRecalculate);
    __global__ void compressCalculateOutput(uint8_t *d_data, size_t size, uint32_t *d_startIndicies, uint32_t *d_startIndiciesLength, uint8_t *d_outputValues, uint8_t *d_outputCounts);

    // Helpers
    void compressCalculateScannedStartMask(uint32_t *d_startMask, uint32_t *d_scannedStartMask, size_t size);

} // RunLength

#endif // RL_GPU_H