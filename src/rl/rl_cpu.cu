#include <stdexcept>

#include "rl_cpu.cuh"

namespace RunLength
{
    CpuCompressed cpuCompress(uint8_t *data, size_t size)
    {
        if (size == 0)
        {
            return CpuCompressed{
                .outputValues = nullptr,
                .outputCounts = nullptr,
                .count = 0,
            };
        }

        // Allocations
        uint8_t *outputValues = reinterpret_cast<uint8_t *>(malloc(sizeof(uint8_t) * size));
        if (outputValues == nullptr)
        {
            throw std::runtime_error("Cannot allocate memory");
        }
        uint8_t *outputCounts = reinterpret_cast<uint8_t *>(malloc(sizeof(uint8_t) * size));
        if (outputCounts == nullptr)
        {
            free(outputValues);
            throw std::runtime_error("Cannot allocate memory");
        }
        size_t count = 0;

        // Compression
        uint8_t currentCount = 1;
        for (size_t i = 1; i < size; i++)
        {
            if (data[i] != data[i - 1] || currentCount == UINT8_MAX)
            {
                outputValues[count] = data[i - 1];
                outputCounts[count] = currentCount;
                currentCount = 0;
                count++;
            }
            currentCount++;
        }
        outputValues[count] = data[size - 1];
        outputCounts[count] = currentCount;
        count++;

        // Reallocate memory to only use as much as needed
        uint8_t *tempValues = reinterpret_cast<uint8_t *>(realloc(outputValues, sizeof(uint8_t) * count));
        if (tempValues == nullptr)
        {
            free(outputValues);
            free(outputCounts);
            throw std::runtime_error("Cannot allocate memory");
        }
        uint8_t *tempCounts = reinterpret_cast<uint8_t *>(realloc(outputCounts, sizeof(uint8_t) * count));
        if (tempCounts == nullptr)
        {
            free(outputValues);
            free(outputCounts);
            throw std::runtime_error("Cannot allocate memory");
        }

        return CpuCompressed{
            .outputValues = outputValues,
            .outputCounts = outputCounts,
            .count = count};
    }
} // RunLength