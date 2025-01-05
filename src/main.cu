#include <cstdio>
#include <cstdint>
#include "./rl/rl_gpu.cuh"

int main(int argc, char **argv)
{

    uint8_t data[] = {5, 5, 8, 8, 8, 7, 7, 7, 7, 3, 4, 4, 4};
    size_t dataSize = 13;

    auto result = RunLength::gpuCompress(data, dataSize);

    for (size_t i = 0; i < result.count; i++)
    {
        printf("%hhu - %hhu\n", result.outputValues[i], result.outputCounts[i]);
    }

    return 0;
}