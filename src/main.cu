#include <iostream>
#include <cstdio>
#include <cstdint>
#include <chrono>

#include "./rl/rl_cpu.cuh"
#include "./rl/rl_gpu.cuh"
#include "./fl/fl_cpu.cuh"

int main(int argc, char **argv)
{
    // uint8_t data[5000000];
    // size_t dataSize = 5000000;
    // uint8_t current = 0;
    // for (size_t i = 1; i <= dataSize; i++)
    // {
    //     data[i - 1] = current;
    //     if (i % 100 == 0)
    //     {
    //         current++;
    //         current %= 100;
    //     }
    // }

    // constexpr size_t dataSize = 500000000;
    // uint8_t *data = reinterpret_cast<uint8_t *>(malloc(sizeof(uint8_t) * dataSize));

    // auto result = RunLength::gpuCompress(data, dataSize);
    // auto result = RunLength::cpuCompress(data, dataSize);

    // printf("size: %llu\n", result.count);

    // auto final = RunLength::gpuDecompress(result.outputValues, result.outputCounts, result.count);
    // auto final = RunLength::cpuDecompress(result.outputValues, result.outputCounts, result.count);

    // printf("final size: %llu\n", final.size);

    // uint8_t counts[] = {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};
    // uint8_t values[] = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
    // size_t size = 16;

    // auto decompressed = RunLength::gpuDecompress(values, counts, size);

    // printf("decompressed size: %llu\n", decompressed.size);

    // for (size_t i = 0; i < decompressed.size; i++)
    // {
    //     printf("%hhu\n", decompressed.data[i]);
    // }

    uint8_t data[130];
    for (size_t i = 0; i < 130; i++)
    {
        data[i] = i % 4;
    }
    size_t size = 130;

    printf("startign\n");

    auto result = FixedLength::cpuCompress(data, size);

    printf("bits size: %llu, values size: %llu\n", result.bitsSize, result.valuesSize);

    for (size_t i = 0; i < result.valuesSize; i++)
    {
        printf("%b\n", result.outputValues[i]);
    }

    return 0;
}