#include "../common.cuh"
#include "../../src/fl/fl_gpu.cuh"

// Compressions
void test_fl_gpu_compression_simple_example(void)
{
    uint8_t data[] = {1, 2, 3, 4, 5, 6, 7};
    size_t size = 7;

    auto result = FixedLength::gpuCompress(data, size);

    size_t expectedBitsCount = 1;
    size_t expectedValuesCount = 3;
    uint8_t expectedBits[] = {3};
    uint8_t expectedValues[] = {0b11010001, 0b01011000, 0b00011111};

    TEST_CHECK_(result.inputSize == size, "%zu is equal to %zu", result.inputSize, size);
    TEST_CHECK_(result.bitsSize == expectedBitsCount, "%zu is equal to %zu", result.bitsSize, expectedBitsCount);
    TEST_CHECK_(result.valuesSize == expectedValuesCount, "%zu is equal to %zu", result.valuesSize, expectedValuesCount);
    TEST_ARRAYS_EQUAL(expectedBits, result.outputBits, expectedBitsCount, "%hhu");
    TEST_ARRAYS_EQUAL(expectedValues, result.outputValues, expectedValuesCount, "%hhu");
}

void test_fl_gpu_compression_all_zeroes(void)
{
    uint8_t data[] = {0, 0, 0, 0, 0, 0, 0, 0};
    size_t size = 8;

    auto result = FixedLength::gpuCompress(data, size);

    size_t expectedBitsCount = 1;
    size_t expectedValuesCount = 1;
    uint8_t expectedBits[] = {1};
    uint8_t expectedValues[] = {0b00000000};

    TEST_CHECK_(result.inputSize == size, "%zu is equal to %zu", result.inputSize, size);
    TEST_CHECK_(result.bitsSize == expectedBitsCount, "%zu is equal to %zu", result.bitsSize, expectedBitsCount);
    TEST_CHECK_(result.valuesSize == expectedValuesCount, "%zu is equal to %zu", result.valuesSize, expectedValuesCount);
    TEST_ARRAYS_EQUAL(expectedBits, result.outputBits, expectedBitsCount, "%hhu");
    TEST_ARRAYS_EQUAL(expectedValues, result.outputValues, expectedValuesCount, "%hhu");
}

void test_fl_gpu_compression_all_ones(void)
{
    uint8_t data[] = {255, 255, 255, 255};
    size_t size = 4;

    auto result = FixedLength::gpuCompress(data, size);

    size_t expectedBitsCount = 1;
    size_t expectedValuesCount = 4;
    uint8_t expectedBits[] = {8};
    uint8_t expectedValues[] = {255, 255, 255, 255};

    TEST_CHECK_(result.inputSize == size, "%zu is equal to %zu", result.inputSize, size);
    TEST_CHECK_(result.bitsSize == expectedBitsCount, "%zu is equal to %zu", result.bitsSize, expectedBitsCount);
    TEST_CHECK_(result.valuesSize == expectedValuesCount, "%zu is equal to %zu", result.valuesSize, expectedValuesCount);
    TEST_ARRAYS_EQUAL(expectedBits, result.outputBits, expectedBitsCount, "%hhu");
    TEST_ARRAYS_EQUAL(expectedValues, result.outputValues, expectedValuesCount, "%hhu");
}

void test_fl_gpu_compression_mixed_values(void)
{
    uint8_t data[] = {128, 64, 32, 16, 8, 4, 2, 1};
    size_t size = 8;

    auto result = FixedLength::gpuCompress(data, size);

    size_t expectedBitsCount = 1;
    size_t expectedValuesCount = 8;
    uint8_t expectedBits[] = {8};
    uint8_t expectedValues[] = {128, 64, 32, 16, 8, 4, 2, 1};

    TEST_CHECK_(result.inputSize == size, "%zu is equal to %zu", result.inputSize, size);
    TEST_CHECK_(result.bitsSize == expectedBitsCount, "%zu is equal to %zu", result.bitsSize, expectedBitsCount);
    TEST_CHECK_(result.valuesSize == expectedValuesCount, "%zu is equal to %zu", result.valuesSize, expectedValuesCount);
    TEST_ARRAYS_EQUAL(expectedBits, result.outputBits, expectedBitsCount, "%hhu");
    TEST_ARRAYS_EQUAL(expectedValues, result.outputValues, expectedValuesCount, "%hhu");
}

void test_fl_gpu_compression_single_value(void)
{
    uint8_t data[] = {42};
    size_t size = 1;

    auto result = FixedLength::gpuCompress(data, size);

    size_t expectedBitsCount = 1;
    size_t expectedValuesCount = 1;
    uint8_t expectedBits[] = {6};
    uint8_t expectedValues[] = {42};

    TEST_CHECK_(result.inputSize == size, "%zu is equal to %zu", result.inputSize, size);
    TEST_CHECK_(result.bitsSize == expectedBitsCount, "%zu is equal to %zu", result.bitsSize, expectedBitsCount);
    TEST_CHECK_(result.valuesSize == expectedValuesCount, "%zu is equal to %zu", result.valuesSize, expectedValuesCount);
    TEST_ARRAYS_EQUAL(expectedBits, result.outputBits, expectedBitsCount, "%hhu");
    TEST_ARRAYS_EQUAL(expectedValues, result.outputValues, expectedValuesCount, "%hhu");
}

void test_fl_gpu_compression_two_frames(void)
{
    uint8_t data[130];
    for (size_t i = 0; i < 130; i++)
    {
        data[i] = i % 4;
    }
    size_t size = 130;

    auto result = FixedLength::gpuCompress(data, size);

    size_t expectedBitsCount = 2;
    size_t expectedValuesCount = 33;
    uint8_t expectedBits[] = {2, 1};
    uint8_t expectedValues[33];

    for (size_t i = 0; i < 32; i++)
    {
        expectedValues[i] = 0b11100100;
    }

    expectedValues[32] = 0b00000010;

    TEST_CHECK_(result.inputSize == size, "%zu is equal to %zu", result.inputSize, size);
    TEST_CHECK_(result.bitsSize == expectedBitsCount, "%zu is equal to %zu", result.bitsSize, expectedBitsCount);
    TEST_CHECK_(result.valuesSize == expectedValuesCount, "%zu is equal to %zu", result.valuesSize, expectedValuesCount);
    TEST_ARRAYS_EQUAL(expectedBits, result.outputBits, expectedBitsCount, "%hhu");
    TEST_ARRAYS_EQUAL(expectedValues, result.outputValues, expectedValuesCount, "%hhu");
}

void test_fl_gpu_compression_three_frames(void)
{
    uint8_t data[384]; // 3 * 128
    for (size_t i = 0; i < 384; i++)
    {
        data[i] = i % 16; // Values 0-15 repeating
    }
    size_t size = 384;

    auto result = FixedLength::gpuCompress(data, size);

    size_t expectedBitsCount = 3;
    size_t expectedValuesCount = 192; // 384 * 4 / 8
    uint8_t expectedBits[] = {4, 4, 4};
    uint8_t expectedValues[192];

    for (size_t i = 0; i < 192; i++)
    {
        switch (i % 8)
        {
        case 0:
            expectedValues[i] = 0b00010000;
            break;
        case 1:
            expectedValues[i] = 0b00110010;
            break;
        case 2:
            expectedValues[i] = 0b01010100;
            break;
        case 3:
            expectedValues[i] = 0b01110110;
            break;
        case 4:
            expectedValues[i] = 0b10011000;
            break;
        case 5:
            expectedValues[i] = 0b10111010;
            break;
        case 6:
            expectedValues[i] = 0b11011100;
            break;
        case 7:
            expectedValues[i] = 0b11111110;
            break;
        }
    }

    TEST_CHECK_(result.inputSize == size, "%zu is equal to %zu", result.inputSize, size);
    TEST_CHECK_(result.bitsSize == expectedBitsCount, "%zu is equal to %zu", result.bitsSize, expectedBitsCount);
    TEST_CHECK_(result.valuesSize == expectedValuesCount, "%zu is equal to %zu", result.valuesSize, expectedValuesCount);
    TEST_ARRAYS_EQUAL(expectedBits, result.outputBits, expectedBitsCount, "%hhu");
    TEST_ARRAYS_EQUAL(expectedValues, result.outputValues, expectedValuesCount, "%hhu");
}

void test_fl_gpu_compression_partial_last_frame(void)
{
    uint8_t data[250];
    for (size_t i = 0; i < 250; i++)
    {
        data[i] = 1;
    }
    size_t size = 250;

    auto result = FixedLength::gpuCompress(data, size);

    size_t expectedBitsCount = 2;
    size_t expectedValuesCount = 32;
    uint8_t expectedBits[] = {1, 1};
    uint8_t expectedValues[32];

    for (size_t i = 0; i < 31; i++)
    {
        expectedValues[i] = 0xFF;
    }
    expectedValues[31] = 0x03;

    TEST_CHECK_(result.inputSize == size, "%zu is equal to %zu", result.inputSize, size);
    TEST_CHECK_(result.bitsSize == expectedBitsCount, "%zu is equal to %zu", result.bitsSize, expectedBitsCount);
    TEST_CHECK_(result.valuesSize == expectedValuesCount, "%zu is equal to %zu", result.valuesSize, expectedValuesCount);
    TEST_ARRAYS_EQUAL(expectedBits, result.outputBits, expectedBitsCount, "%hhu");
    TEST_ARRAYS_EQUAL(expectedValues, result.outputValues, expectedValuesCount, "%hhu");
}

void test_fl_gpu_compression_different_bits_per_frame(void)
{
    uint8_t data[256]; // 2 full frames
    for (size_t i = 0; i < 128; i++)
    {
        data[i] = 6; // 110
    }
    for (size_t i = 128; i < 256; i++)
    {
        data[i] = 11; // 1011
    }
    size_t size = 256;

    auto result = FixedLength::gpuCompress(data, size);

    size_t expectedBitsCount = 2;
    size_t expectedValuesCount = 112;
    uint8_t expectedBits[] = {3, 4};
    uint8_t expectedValues[112];

    for (size_t i = 0; i < 48; i += 3)
    {
        expectedValues[i] = 0b10110110;
        if (i + 1 < 48)
        {
            expectedValues[i + 1] = 0b01101101;
        }
        if (i + 2 < 48)
        {
            expectedValues[i + 2] = 0b11011011;
        }
    }

    for (size_t i = 48; i < 112; i++)
    {
        expectedValues[i] = 0b10111011;
    }

    TEST_CHECK_(result.inputSize == size, "%zu is equal to %zu", result.inputSize, size);
    TEST_CHECK_(result.bitsSize == expectedBitsCount, "%zu is equal to %zu", result.bitsSize, expectedBitsCount);
    TEST_CHECK_(result.valuesSize == expectedValuesCount, "%zu is equal to %zu", result.valuesSize, expectedValuesCount);
    TEST_ARRAYS_EQUAL(expectedBits, result.outputBits, expectedBitsCount, "%hhu");
    TEST_ARRAYS_EQUAL(expectedValues, result.outputValues, expectedValuesCount, "%hhu");
}

TEST_LIST = {
    // Compressions
    {"test_fl_gpu_compression_simple_example", test_fl_gpu_compression_simple_example},
    {"test_fl_gpu_compression_all_zeroes", test_fl_gpu_compression_all_zeroes},
    {"test_fl_gpu_compression_all_ones", test_fl_gpu_compression_all_ones},
    {"test_fl_gpu_compression_mixed_values", test_fl_gpu_compression_mixed_values},
    {"test_fl_gpu_compression_single_value", test_fl_gpu_compression_single_value},
    {"test_fl_gpu_compression_two_frames", test_fl_gpu_compression_two_frames},
    {"test_fl_gpu_compression_three_frames", test_fl_gpu_compression_three_frames},
    {"test_fl_gpu_compression_partial_last_frame", test_fl_gpu_compression_partial_last_frame},
    {"test_fl_gpu_compression_different_bits_per_frame", test_fl_gpu_compression_different_bits_per_frame},
    {nullptr, nullptr}};