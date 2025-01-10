#ifndef FL_CPU_H
#define FL_CPU_H

#include "fl_common.cuh"

namespace FixedLength
{
    FLCompressed cpuCompress(uint8_t *data, size_t size);
    FLDecompressed cpuDecompress(size_t outputSize, uint8_t *bits, size_t bitsSize, uint8_t *values, size_t valuesSize);
}

#endif // FL_CPU_H