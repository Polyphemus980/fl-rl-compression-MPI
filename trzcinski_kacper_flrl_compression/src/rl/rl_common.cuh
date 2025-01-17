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

        RLCompressed()
        {
            outputValues = nullptr;
            outputCounts = nullptr;
            count = 0;
        };

        RLCompressed(uint8_t *outputValues, uint8_t *outputCounts, size_t count)
        {
            this->outputValues = outputValues;
            this->outputCounts = outputCounts;
            this->count = count;
        };
    };

    struct RLDecompressed
    {
        uint8_t *data;
        size_t size;

        RLDecompressed()
        {
            this->data = nullptr;
            this->size = 0;
        };

        RLDecompressed(uint8_t *data, size_t size)
        {
            this->data = data;
            this->size = size;
        };
    };
} // RunLength

#endif // RL_COMMON_H
