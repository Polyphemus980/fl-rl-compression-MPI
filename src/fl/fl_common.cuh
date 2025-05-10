#ifndef FL_COMMON_H
#define FL_COMMON_H

#include <cstdint>
namespace FixedLength
{
    // Number of bytes per frame
    static constexpr size_t FRAME_LENGTH = 128;

    struct FLCompressed
    {
        uint8_t *outputBits;
        size_t bitsSize;
        uint8_t *outputValues;
        size_t valuesSize;
        size_t inputSize;

        FLCompressed()
        {
            this->outputBits = nullptr;
            this->bitsSize = 0;
            this->outputValues = nullptr;
            this->valuesSize = 0;
            this->inputSize = 0;
        };
        FLCompressed(uint8_t *outputBits, size_t bitsSize, uint8_t *outputValues, size_t valuesSize, size_t inputSize)
        {
            this->outputBits = outputBits;
            this->bitsSize = bitsSize;
            this->outputValues = outputValues;
            this->valuesSize = valuesSize;
            this->inputSize = inputSize;
        }
    };

    struct FLDecompressed
    {
        uint8_t *data;
        size_t size;

        FLDecompressed()
        {
            this->data = nullptr;
            this->size = 0;
        };

        FLDecompressed(uint8_t *data, size_t size)
        {
            this->data = data;
            this->size = size;
        };
    };

    inline __device__ __host__ uint8_t countLeadingZeroes(uint8_t value)
    {
        if (value == 0)
        {
            return 8;
        }
        uint8_t count = 0;
        uint8_t mask = 1 << 7;
        while (!(value & mask))
        {
            count++;
            value <<= 1;
        }
        return count;
    }
} //  FixedLength

#endif // FL_COMMON_H