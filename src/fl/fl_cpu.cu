#include <cmath>
#include <stdexcept>

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

        // Allocate bits array
        const size_t framesCount = ceil(size * 1.0 / FRAME_LENGTH);

        uint8_t *outputBits = reinterpret_cast<uint8_t *>(malloc(sizeof(uint8_t) * framesCount));

        if (outputBits == nullptr)
        {
            throw std::runtime_error("Cannot allocate memory\n");
        }

        // Calculate outputBits
        size_t totalBitsRequired = 0;
        for (size_t f = 0; f < framesCount; f++)
        {
            // We set it to 1 so that when we have 0 (which in our computation returns requiredBits = 0)
            // we don't have to manually adjust it every time that it actually need 1 bit.
            uint8_t minBits = 1;
            for (size_t i = 0; i < FRAME_LENGTH && f * FRAME_LENGTH + i < size; i++)
            {
                uint8_t leadingZeroes = countLeadingZeroes(data[f * FRAME_LENGTH + i]);
                uint8_t requiredBits = 8 - leadingZeroes;
                minBits = max(minBits, requiredBits);
            }
            outputBits[f] = minBits;
            totalBitsRequired += minBits * min(FRAME_LENGTH, size - FRAME_LENGTH * f);
        }

        // Allocate values array
        const size_t valuesSize = ceil(totalBitsRequired * 1.0 / 8);
        uint8_t *outputValues = reinterpret_cast<uint8_t *>(malloc(sizeof(uint8_t) * valuesSize));
        memset(outputValues, 0, sizeof(uint8_t) * valuesSize);

        // Compression
        size_t usedBits = 0;
        for (size_t f = 0; f < framesCount; f++)
        {
            uint8_t requiredBits = outputBits[f];
            for (size_t i = 0; i < FRAME_LENGTH && f * FRAME_LENGTH + i < size; i++)
            {
                uint8_t value = data[f * FRAME_LENGTH + i];
                size_t outputId = usedBits / 8;
                uint8_t outputOffset = usedBits % 8;
                // Encode value
                uint8_t encodedValue = value << outputOffset;
                outputValues[outputId] |= encodedValue;

                // If it overflows encode the overflowed part on next byte
                if (outputOffset + requiredBits > 8)
                {
                    uint8_t overflowValue = value >> (8 - outputOffset);
                    outputValues[outputId + 1] |= overflowValue;
                }
                usedBits += requiredBits;
            }
        }

        return FLCompressed{
            .outputBits = outputBits,
            .bitsSize = framesCount,
            .outputValues = outputValues,
            .valuesSize = valuesSize};
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

    uint8_t countLeadingZeroes(uint8_t value)
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

} // FixedLength