#ifndef RL_CPU_H
#define RL_CPU_H

#include "rl_common.cuh"

namespace RunLength
{
    CpuCompressed cpuCompress(uint8_t *data, size_t size);
    CpuDecompressed cpuDecompress(uint8_t *values, uint8_t *counts, size_t size);
} // RunLength

#endif // RL_CPU_H