#ifndef RL_CPU_H
#define RL_CPU_H

#include "rl_common.cuh"

namespace RunLength
{
    CpuCompressed cpuCompress(uint8_t *data, size_t size);
} // RunLength

#endif // RL_CPU_H