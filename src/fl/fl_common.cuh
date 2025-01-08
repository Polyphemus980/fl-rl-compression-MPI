#ifndef FL_COMMON_H
#define FL_COMMON_H

#include <cstdint>

namespace FixedLength
{
    // Number of bytes per frame
    static constexpr size_t FRAME_LENGTH = 16;

    struct FLCompressed
    {
        uint8_t *outputBits;
        size_t bitsSize;
        uint8_t *outputValues;
        size_t valuesSize;
    };

    struct FLDecompressed
    {
        uint8_t *data;
        size_t size;
    };
} //  FixedLength

#endif // FL_COMMON_H