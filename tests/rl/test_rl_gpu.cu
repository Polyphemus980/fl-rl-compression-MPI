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

void test_rl_gpu_compression_large_sequence_2(void)
{
    uint8_t data[2550];
    size_t dataSize = 2550;

    for (size_t i = 0; i < dataSize; ++i)
    {
        data[i] = 100;
    }

    uint8_t expectedCounts[] = {255, 255, 255, 255, 255, 255, 255, 255, 255, 255};
    uint8_t expectedValues[] = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
    size_t expectedCount = 10;

    auto compressedData = RunLength::gpuCompress(data, dataSize);
    TEST_CHECK_(compressedData.count == expectedCount, "%zu is equal to %zu", compressedData.count, expectedCount);
    TEST_ARRAYS_EQUAL(expectedCounts, compressedData.outputCounts, expectedCount, "%hhu");
    TEST_ARRAYS_EQUAL(expectedValues, compressedData.outputValues, expectedCount, "%hhu");
}

void test_rl_gpu_compression_large_sequence_3(void)
{
    uint8_t data[512];
    size_t dataSize = 512;

    data[0] = 0;
    for (size_t i = 0; i < 510; i++)
    {
        data[i + 1] = 1;
    }

    data[511] = 2;

    uint8_t expectedCounts[] = {1, 255, 255, 1};
    uint8_t expectedValues[] = {0, 1, 1, 2};
    size_t expectedCount = 4;

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

// Decompressions
void test_rl_gpu_decompression_implementation_plan_example(void)
{

    uint8_t counts[] = {2, 3, 4, 1, 3};
    uint8_t values[] = {5, 8, 7, 3, 4};
    size_t size = 5;

    uint8_t expectedData[] = {5, 5, 8, 8, 8, 7, 7, 7, 7, 3, 4, 4, 4};
    size_t expecetedSize = 13;
    auto decompressedData = RunLength::gpuDecompress(values, counts, size);
    TEST_CHECK_(decompressedData.size == expecetedSize, "%zu is equal to %zu", decompressedData.size, expecetedSize);
    TEST_ARRAYS_EQUAL(expectedData, decompressedData.data, expecetedSize, "%hhu");
}

void test_rl_gpu_decompression_empty(void)
{
    uint8_t counts[] = {};
    uint8_t values[] = {};
    size_t size = 0;

    uint8_t expectedData[] = {};
    size_t expectedSize = 0;

    auto decompressedData = RunLength::gpuDecompress(values, counts, size);
    TEST_CHECK_(decompressedData.size == expectedSize, "%zu is equal to %zu", decompressedData.size, expectedSize);
    TEST_ARRAYS_EQUAL(expectedData, decompressedData.data, expectedSize, "%hhu");
}

void test_rl_gpu_decompression_single_element(void)
{
    uint8_t counts[] = {5};
    uint8_t values[] = {9};
    size_t size = 1;

    uint8_t expectedData[] = {9, 9, 9, 9, 9};
    size_t expectedSize = 5;

    auto decompressedData = RunLength::gpuDecompress(values, counts, size);
    TEST_CHECK_(decompressedData.size == expectedSize, "%zu is equal to %zu", decompressedData.size, expectedSize);
    TEST_ARRAYS_EQUAL(expectedData, decompressedData.data, expectedSize, "%hhu");
}

void test_rl_gpu_decompression_alternating_values(void)
{
    uint8_t counts[] = {1, 2, 1, 3};
    uint8_t values[] = {1, 2, 3, 4};
    size_t size = 4;

    uint8_t expectedData[] = {1, 2, 2, 3, 4, 4, 4};
    size_t expectedSize = 7;

    auto decompressedData = RunLength::gpuDecompress(values, counts, size);
    TEST_CHECK_(decompressedData.size == expectedSize, "%zu is equal to %zu", decompressedData.size, expectedSize);
    TEST_ARRAYS_EQUAL(expectedData, decompressedData.data, expectedSize, "%hhu");
}

void test_rl_gpu_decompression_large_sequence(void)
{
    uint8_t counts[] = {255};
    uint8_t values[] = {100};
    size_t size = 1;

    uint8_t expectedData[255];
    for (size_t i = 0; i < 255; ++i)
    {
        expectedData[i] = 100;
    }
    size_t expectedSize = 255;

    auto decompressedData = RunLength::gpuDecompress(values, counts, size);
    TEST_CHECK_(decompressedData.size == expectedSize, "%zu is equal to %zu", decompressedData.size, expectedSize);
    TEST_ARRAYS_EQUAL(expectedData, decompressedData.data, expectedSize, "%hhu");
}

void test_rl_gpu_decompression_same_element_with_count_over_255(void)
{
    uint8_t counts[] = {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};
    uint8_t values[] = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
    size_t size = 16;

    uint8_t expectedData[16 * 255];
    for (size_t i = 0; i < 16 * 255; i++)
    {
        expectedData[i] = 100;
    }
    size_t expectedSize = 16 * 255;

    auto decompressedData = RunLength::gpuDecompress(values, counts, size);
    TEST_CHECK_(decompressedData.size == expectedSize, "%zu is equal to %zu", decompressedData.size, expectedSize);
    TEST_ARRAYS_EQUAL(expectedData, decompressedData.data, expectedSize, "%hhu");
}

void test_rl_gpu_decompression_zero_count(void)
{
    uint8_t counts[] = {0, 3, 2};
    uint8_t values[] = {9, 5, 8};
    size_t size = 3;

    uint8_t expectedData[] = {5, 5, 5, 8, 8};
    size_t expectedSize = 5;

    auto decompressedData = RunLength::gpuDecompress(values, counts, size);
    TEST_CHECK_(decompressedData.size == expectedSize, "%zu is equal to %zu", decompressedData.size, expectedSize);
    TEST_ARRAYS_EQUAL(expectedData, decompressedData.data, expectedSize, "%hhu");
}

// TODO: add tests that do both compression and then decompression

TEST_LIST = {
    // Compression
    {"test_rl_gpu_compression_implementation_plan_example", test_rl_gpu_compression_implementation_plan_example},
    {"test_rl_gpu_compression_empty", test_rl_gpu_compression_empty},
    {"test_rl_gpu_compression_single_value", test_rl_gpu_compression_single_value},
    {"test_rl_gpu_compression_single_sequence", test_rl_gpu_compression_single_sequence},
    {"test_rl_gpu_compression_unique_elements", test_rl_gpu_compression_unique_elements},
    {"test_rl_gpu_compression_large_sequence", test_rl_gpu_compression_large_sequence},
    {"test_rl_gpu_compression_large_sequence_2", test_rl_gpu_compression_large_sequence_2},
    {"test_rl_gpu_compression_large_sequence_3", test_rl_gpu_compression_large_sequence_3},
    {"test_rl_gpu_compression_more_than_one_block", test_rl_gpu_compression_more_than_one_block},
    {"test_rl_gpu_compression_huge_data", test_rl_gpu_compression_huge_data},
    // Decompression
    {"test_rl_gpu_decompression_implementation_plan_example", test_rl_gpu_decompression_implementation_plan_example},
    {"test_rl_gpu_decompression_empty", test_rl_gpu_decompression_empty},
    {"test_rl_gpu_decompression_single_element", test_rl_gpu_decompression_single_element},
    {"test_rl_gpu_decompression_alternating_values", test_rl_gpu_decompression_alternating_values},
    {"test_rl_gpu_decompression_large_sequence", test_rl_gpu_decompression_large_sequence},
    {"test_rl_gpu_decompression_same_element_with_count_over_255", test_rl_gpu_decompression_same_element_with_count_over_255},
    {"test_rl_gpu_decompression_zero_count", test_rl_gpu_decompression_zero_count},
    {nullptr, nullptr}};