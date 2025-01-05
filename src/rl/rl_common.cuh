#ifndef RL_COMMON_H
#define RL_COMMON_H

#include <cstdint>

namespace RunLength
{
    struct RLCompressed
    {
        uint8_t *outputValues;
        // uint8_t is in range up to 255 which should be enough in most cases
        // In case if it's actually not enough we split the sequence into
        // two sequences next to each other.
        uint8_t *outputCounts;
        size_t count;
    };

    struct RLDecompressed
    {
        uint8_t *data;
        size_t size;
    };
} // RunLength

#endif // RL_COMMON_H
