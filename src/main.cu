#include <iostream>
#include <cstdio>
#include <cstdint>
#include <chrono>

#include "./rl/rl_cpu.cuh"
#include "./rl/rl_gpu.cuh"

int main(int argc, char **argv)
{
    // auto start = std::chrono::high_resolution_clock::now();

    // uint8_t data[256];
    // size_t dataSize = 256;

    // for (size_t i = 0; i < dataSize; ++i)
    // {
    //     data[i] = 100;
    // }

    // auto result = RunLength::gpuCompress(data, dataSize);
    // // auto result = RunLength::cpuCompress(data, dataSize);

    // auto end = std::chrono::high_resolution_clock::now();

    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // std::cout << "Time taken for all: " << duration.count() << " ms" << std::endl;

    // for (size_t i = 0; i < result.count; i++)
    // {
    //     printf("%hhu - %hhu\n", result.outputValues[i], result.outputCounts[i]);
    // }

    uint8_t counts[] = {2, 3, 4, 1, 3};
    uint8_t values[] = {5, 8, 7, 3, 4};
    size_t size = 5;

    auto decompressed = RunLength::gpuDecompress(values, counts, size);

    for (size_t i = 0; i < decompressed.size; i++)
    {
        printf("%hhu\n", decompressed.data[i]);
    }

    return 0;
}