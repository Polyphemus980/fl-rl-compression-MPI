#include "../common.cuh"
#include "../../src/rl/rl_gpu.cuh"

// Compressions
void test_rl_gpu_compression_implementation_plan_example(void)
{
    uint8_t data[] = {5, 5, 8, 8, 8, 7, 7, 7, 7, 3, 4, 4, 4};
    size_t dataSize = 13;
    uint8_t expectedCounts[] = {2, 3, 4, 1, 3};
    uint8_t expectedValues[] = {5, 8, 7, 3, 4};
    size_t expectedCount = 5;

    auto compressedData = RunLength::gpuCompress(data, dataSize);
    TEST_CHECK_(compressedData.count == expectedCount, "%zu is equal to %zu", compressedData.count, expectedCount);
    TEST_ARRAYS_EQUAL(expectedCounts, compressedData.outputCounts, expectedCount, "%hhu");
    TEST_ARRAYS_EQUAL(expectedValues, compressedData.outputValues, expectedCount, "%hhu");
}

void test_rl_gpu_compression_empty(void)
{
    uint8_t data[] = {};
    size_t dataSize = 0;
    uint8_t expectedCounts[] = {};
    uint8_t expectedValues[] = {};
    size_t expectedCount = 0;

    auto compressedData = RunLength::gpuCompress(data, dataSize);
    TEST_CHECK_(compressedData.count == expectedCount, "%zu is equal to %zu", compressedData.count, expectedCount);
    TEST_ARRAYS_EQUAL(expectedCounts, compressedData.outputCounts, expectedCount, "%hhu");
    TEST_ARRAYS_EQUAL(expectedValues, compressedData.outputValues, expectedCount, "%hhu");
}

void test_rl_gpu_compression_single_value(void)
{
    uint8_t data[] = {9};
    size_t dataSize = 1;
    uint8_t expectedCounts[] = {1};
    uint8_t expectedValues[] = {9};
    size_t expectedCount = 1;

    auto compressedData = RunLength::gpuCompress(data, dataSize);
    TEST_CHECK_(compressedData.count == expectedCount, "%zu is equal to %zu", compressedData.count, expectedCount);
    TEST_ARRAYS_EQUAL(expectedCounts, compressedData.outputCounts, expectedCount, "%hhu");
    TEST_ARRAYS_EQUAL(expectedValues, compressedData.outputValues, expectedCount, "%hhu");
}

void test_rl_gpu_compression_single_sequence(void)
{
    uint8_t data[] = {9, 9, 9, 9, 9};
    size_t dataSize = 5;
    uint8_t expectedCounts[] = {5};
    uint8_t expectedValues[] = {9};
    size_t expectedCount = 1;

    auto compressedData = RunLength::gpuCompress(data, dataSize);
    TEST_CHECK_(compressedData.count == expectedCount, "%zu is equal to %zu", compressedData.count, expectedCount);
    TEST_ARRAYS_EQUAL(expectedCounts, compressedData.outputCounts, expectedCount, "%hhu");
    TEST_ARRAYS_EQUAL(expectedValues, compressedData.outputValues, expectedCount, "%hhu");
}

void test_rl_gpu_compression_unique_elements(void)
{
    uint8_t data[] = {1, 2, 3, 4, 5};
    size_t dataSize = 5;
    uint8_t expectedCounts[] = {1, 1, 1, 1, 1};
    uint8_t expectedValues[] = {1, 2, 3, 4, 5};
    size_t expectedCount = 5;

    auto compressedData = RunLength::gpuCompress(data, dataSize);
    TEST_CHECK_(compressedData.count == expectedCount, "%zu is equal to %zu", compressedData.count, expectedCount);
    TEST_ARRAYS_EQUAL(expectedCounts, compressedData.outputCounts, expectedCount, "%hhu");
    TEST_ARRAYS_EQUAL(expectedValues, compressedData.outputValues, expectedCount, "%hhu");
}

void test_rl_gpu_compression_large_sequence(void)
{
    uint8_t data[256];
    size_t dataSize = 256;

    for (size_t i = 0; i < dataSize; ++i)
    {
        data[i] = 100;
    }

    uint8_t expectedCounts[] = {255, 1};
    uint8_t expectedValues[] = {100, 100};
    size_t expectedCount = 2;

    auto compressedData = RunLength::gpuCompress(data, dataSize);
    TEST_CHECK_(compressedData.count == expectedCount, "%zu is equal to %zu", compressedData.count, expectedCount);
    TEST_ARRAYS_EQUAL(expectedCounts, compressedData.outputCounts, expectedCount, "%hhu");
    TEST_ARRAYS_EQUAL(expectedValues, compressedData.outputValues, expectedCount, "%hhu");
}

void test_rl_gpu_compression_more_than_one_block(void)
{
    uint8_t data[5000];
    size_t dataSize = 5000;
    uint8_t current = 0;
    for (size_t i = 1; i <= dataSize; i++)
    {
        data[i - 1] = current;
        if (i % 50 == 0)
        {
            current++;
        }
    }

    uint8_t expectedCounts[100];
    uint8_t expectedValues[100];
    size_t expectedCount = 100;

    for (size_t i = 0; i < expectedCount; i++)
    {
        expectedCounts[i] = 50;
        expectedValues[i] = i;
    }

    auto compressedData = RunLength::gpuCompress(data, dataSize);
    TEST_CHECK_(compressedData.count == expectedCount, "%zu is equal to %zu", compressedData.count, expectedCount);
    TEST_ARRAYS_EQUAL(expectedCounts, compressedData.outputCounts, expectedCount, "%hhu");
    TEST_ARRAYS_EQUAL(expectedValues, compressedData.outputValues, expectedCount, "%hhu");
}

void test_rl_gpu_compression_huge_data(void)
{
    uint8_t data[5000000];
    size_t dataSize = 5000000;
    uint8_t current = 0;
    for (size_t i = 1; i <= dataSize; i++)
    {
        data[i - 1] = current;
        if (i % 100 == 0)
        {
            current++;
            current %= 100;
        }
    }

    uint8_t expectedCounts[50000];
    uint8_t expectedValues[50000];
    size_t expectedCount = 50000;

    for (size_t i = 0; i < expectedCount; i++)
    {
        expectedCounts[i] = 100;
        expectedValues[i] = (i % 100);
    }

    auto compressedData = RunLength::gpuCompress(data, dataSize);
    TEST_CHECK_(compressedData.count == expectedCount, "%zu is equal to %zu", compressedData.count, expectedCount);
    TEST_ARRAYS_EQUAL(expectedCounts, compressedData.outputCounts, expectedCount, "%hhu");
    TEST_ARRAYS_EQUAL(expectedValues, compressedData.outputValues, expectedCount, "%hhu");
}

TEST_LIST = {
    // Compression
    {"test_rl_gpu_compression_implementation_plan_example", test_rl_gpu_compression_implementation_plan_example},
    {"test_rl_gpu_compression_empty", test_rl_gpu_compression_empty},
    {"test_rl_gpu_compression_single_value", test_rl_gpu_compression_single_value},
    {"test_rl_gpu_compression_single_sequence", test_rl_gpu_compression_single_sequence},
    {"test_rl_gpu_compression_unique_elements", test_rl_gpu_compression_unique_elements},
    {"test_rl_gpu_compression_large_sequence", test_rl_gpu_compression_large_sequence},
    {"test_rl_gpu_compression_more_than_one_block", test_rl_gpu_compression_more_than_one_block},
    {"test_rl_gpu_compression_huge_data", test_rl_gpu_compression_huge_data},
    {nullptr, nullptr}};