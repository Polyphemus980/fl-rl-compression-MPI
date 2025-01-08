#include "fl_cpu.cuh"

namespace FixedLength
{
    FLCompressed cpuCompress(uint8_t *data, size_t size)
    {
        if (size == 0)
        {
            return FLCompressed{
                .outputBits = nullptr,
                .bitsSize = 0,
                .outputValues = nullptr,
                .valuesSize = 0};
        }

        // TODO:
    }

    FLDecompressed cpuDecompress(uint8_t *bits, size_t bitsSize, uint8_t *values, size_t valuesSize)
    {
        if (valuesSize == 0 || bitsSize == 0)
        {
            return FLDecompressed{
                .data = nullptr,
                .size = 0};
        }

        // TODO:
    }

} // FixedLength