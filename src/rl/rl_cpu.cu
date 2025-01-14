#include <stdexcept>

#include "rl_cpu.cuh"
#include "../timers/cpu_timer.cuh"

namespace RunLength
{
    RLCompressed cpuCompress(uint8_t *data, size_t size)
    {
        if (size == 0)
        {
            return RLCompressed();
        }

        Timers::CpuTimer cpuTimer;

        cpuTimer.start();

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

        cpuTimer.end();
        cpuTimer.printResult("Allocate arrays on CPU");

        cpuTimer.start();

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

        cpuTimer.end();
        cpuTimer.printResult("Compression");

        cpuTimer.start();

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

        cpuTimer.end();
        cpuTimer.printResult("Reallocate arrays on CPU");

        return RLCompressed(outputValues, outputCounts, count);
    }

    RLDecompressed cpuDecompress(uint8_t *values, uint8_t *counts, size_t size)
    {
        if (size == 0)
        {
            return RLDecompressed();
        }

        Timers::CpuTimer cpuTimer;

        cpuTimer.start();

        size_t outputSize = 0;
        for (size_t i = 0; i < size; i++)
        {
            outputSize += counts[i];
        }

        // Allocation
        uint8_t *data = reinterpret_cast<uint8_t *>(malloc(sizeof(uint8_t) * outputSize));
        if (data == nullptr)
        {
            throw std::runtime_error("Cannot allocate memory");
        }

        // Decompression
        size_t global_id = 0;
        for (size_t i = 0; i < size; i++)
        {
            for (size_t j = 0; j < counts[i]; j++)
            {
                data[global_id++] = values[i];
            }
        }

        cpuTimer.end();
        cpuTimer.printResult("Decompression");

        return RLDecompressed(data, outputSize);
    }
} // RunLength