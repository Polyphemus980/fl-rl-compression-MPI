#include <cstdio>
#include <cstdint>
#include "./rl/rl_gpu.cuh"

int main(int argc, char **argv)
{

    uint8_t data[] = {9, 9, 9, 9, 9};
    size_t dataSize = 5;

    auto result = RunLength::gpuCompress(data, dataSize);

    for (size_t i = 0; i < result.count; i++)
    {
        printf("%hhu - %hhu\n", result.outputValues[i], result.outputCounts[i]);
    }

    return 0;
}