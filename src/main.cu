#include <iostream>
#include <cstdio>
#include <cstdint>
#include <chrono>

#include "./rl/rl_cpu.cuh"
#include "./rl/rl_gpu.cuh"
#include "./fl/fl_cpu.cuh"
#include "./fl/fl_gpu.cuh"

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

    // uint8_t data[130];
    // for (size_t i = 0; i < 130; i++)
    // {
    //     data[i] = i % 4;
    // }
    // size_t size = 130;

    // printf("startign\n");

    // auto result = FixedLength::cpuCompress(data, size);

    // printf("bits size: %llu, values size: %llu\n", result.bitsSize, result.valuesSize);

    // size_t bitsCount = 2;
    // size_t valuesCount = 112; // (128 * 3 + 128 * 4) / 8
    // uint8_t bits[] = {3, 4};

    // uint8_t values[112] = {0};

    // // Only 6s
    // for (size_t i = 0; i < 48; i += 3)
    // {
    //     values[i] = 0b10110110;
    //     values[i + 1] = 0b11011001;
    //     values[i + 2] = 0b01101101;
    // }

    // // Only 11s
    // for (size_t i = 48; i < 112; i++)
    // {
    //     values[i] = 0b10111011;
    // }

    // size_t outputSize = 256;

    // auto result2 = FixedLength::cpuDecompress(outputSize, bits, bitsCount, values, valuesCount);

    // for (size_t i = 0; i < result.size; i++)
    // {
    //     printf("%hhu\n", result.data[i]);
    // }

    // constexpr size_t size0 = 1;
    // uint8_t data0[size0];
    // data0[0] = 0;

    // auto result0 = FixedLength::gpuCompress(data0, size0);
    // printf("\n\n");

    // constexpr size_t size = 1025;
    // uint8_t data[size];
    // for (size_t i = 0; i < 1024; i++)
    // {
    //     data[i] = 8;
    // }
    // data[1024] = 127;

    // auto result = FixedLength::gpuCompress(data, size);
    // printf("\n\n");

    // constexpr size_t size2 = 1025;
    // uint8_t data2[size];
    // uint8_t value = 1;
    // for (size_t i = 0; i < 1024; i++)
    // {
    //     data2[i] = value;
    //     if (i % 128 == 0 && i > 0)
    //     {
    //         value *= 2;
    //     }
    // }
    // data2[1024] = 0;

    // auto result2 = FixedLength::gpuCompress(data2, size2);
    // printf("\n\n");

    constexpr size_t size = 10000;
    uint8_t data[size];

    for (size_t i = 0; i < size; i++)
    {
        data[i] = rand() % 256;
    }

    auto result = FixedLength::gpuCompress(data, size);
    printf("result size: %lu\n", result.valuesSize);

    for (size_t i = 0; i < result.valuesSize; i++)
    {
        printf("%b\n", result.outputValues[i]);
    }

    auto decompressed = FixedLength::gpuDecompress(result.inputSize, result.outputBits, result.bitsSize, result.outputValues, result.valuesSize);

    if (size != decompressed.size)
    {
        printf("XD\n");
    }

    for (size_t i = 0; i < size; i++)
    {
        if (data[i] != decompressed.data[i])
        {
            printf("wrong at %d\n", i);
        }
    }

    printf("actually ok\n");

    // size_t bitsCount = 1;
    // size_t valuesCount = 3;
    // uint8_t bits[] = {3};
    // uint8_t values[] = {0b11010001, 0b01011000, 0b00011111};
    // size_t outputSize = 7;

    // auto result = FixedLength::gpuDecompress(outputSize, bits, bitsCount, values, valuesCount);

    // printf("SIZE: %llu\n", result.size);
    // for (size_t i = 0; i < result.size; i++)
    // {
    //     printf("value: %hhu\n", result.data[i]);
    // }

    return 0;
}