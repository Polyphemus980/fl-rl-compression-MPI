#include "../common.cuh"
#include "../../src/fl/fl_cpu.cuh"

// Compressions
void test_fl_cpu_compression_simple_example(void)
{
    uint8_t data[] = {1, 2, 3, 4, 5, 6, 7};
    size_t size = 7;

    auto result = FixedLength::cpuCompress(data, size);

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

void test_fl_cpu_compression_all_zeroes(void)
{
    uint8_t data[] = {0, 0, 0, 0, 0, 0, 0, 0};
    size_t size = 8;

    auto result = FixedLength::cpuCompress(data, size);

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

void test_fl_cpu_compression_all_ones(void)
{
    uint8_t data[] = {255, 255, 255, 255};
    size_t size = 4;

    auto result = FixedLength::cpuCompress(data, size);

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

void test_fl_cpu_compression_mixed_values(void)
{
    uint8_t data[] = {128, 64, 32, 16, 8, 4, 2, 1};
    size_t size = 8;

    auto result = FixedLength::cpuCompress(data, size);

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

void test_fl_cpu_compression_single_value(void)
{
    uint8_t data[] = {42};
    size_t size = 1;

    auto result = FixedLength::cpuCompress(data, size);

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

void test_fl_cpu_compression_two_frames(void)
{
    uint8_t data[130];
    for (size_t i = 0; i < 130; i++)
    {
        data[i] = i % 4;
    }
    size_t size = 130;

    auto result = FixedLength::cpuCompress(data, size);

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

void test_fl_cpu_compression_three_frames(void)
{
    uint8_t data[384]; // 3 * 128
    for (size_t i = 0; i < 384; i++)
    {
        data[i] = i % 16; // Values 0-15 repeating
    }
    size_t size = 384;

    auto result = FixedLength::cpuCompress(data, size);

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

void test_fl_cpu_compression_partial_last_frame(void)
{
    uint8_t data[250];
    for (size_t i = 0; i < 250; i++)
    {
        data[i] = 1;
    }
    size_t size = 250;

    auto result = FixedLength::cpuCompress(data, size);

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

void test_fl_cpu_compression_different_bits_per_frame(void)
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

    auto result = FixedLength::cpuCompress(data, size);

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

// Decompressions
void test_fl_cpu_decompression_simple_example(void)
{
    size_t bitsCount = 1;
    size_t valuesCount = 3;
    uint8_t bits[] = {3};
    uint8_t values[] = {0b11010001, 0b01011000, 0b00011111};
    size_t outputSize = 7;

    auto result = FixedLength::cpuDecompress(outputSize, bits, bitsCount, values, valuesCount);

    uint8_t expectedData[] = {1, 2, 3, 4, 5, 6, 7};
    size_t expectedSize = 7;

    TEST_CHECK_(result.size == expectedSize, "%zu is equal to %zu", result.size, expectedSize);
    TEST_ARRAYS_EQUAL(expectedData, result.data, expectedSize, "%hhu");
}

void test_fl_cpu_decompression_single_element(void)
{
    size_t bitsCount = 1;
    size_t valuesCount = 1;
    uint8_t bits[] = {8};
    uint8_t values[] = {0b11111111};
    size_t outputSize = 1;

    auto result = FixedLength::cpuDecompress(outputSize, bits, bitsCount, values, valuesCount);

    uint8_t expectedData[] = {255};
    size_t expectedSize = 1;

    TEST_CHECK_(result.size == expectedSize, "%zu is equal to %zu", result.size, expectedSize);
    TEST_ARRAYS_EQUAL(expectedData, result.data, expectedSize, "%hhu");
}

void test_fl_cpu_decompression_single_frame_partial(void)
{
    size_t bitsCount = 1;
    size_t valuesCount = 3;
    uint8_t bits[] = {4};
    uint8_t values[] = {0b00111010, 0b11001111, 0b00001000};
    size_t outputSize = 5;

    auto result = FixedLength::cpuDecompress(outputSize, bits, bitsCount, values, valuesCount);

    uint8_t expectedData[] = {10, 3, 15, 12, 8};
    size_t expectedSize = 5;

    TEST_CHECK_(result.size == expectedSize, "%zu is equal to %zu", result.size, expectedSize);
    TEST_ARRAYS_EQUAL(expectedData, result.data, expectedSize, "%hhu");
}

void test_fl_cpu_decompression_two_frames(void)
{
    size_t bitsCount = 2;
    size_t valuesCount = 112; // (128 * 3 + 128 * 4) / 8
    uint8_t bits[] = {3, 4};

    uint8_t values[112] = {0};

    // Only 6s
    for (size_t i = 0; i < 48; i += 3)
    {
        values[i] = 0b10110110;
        if (i + 1 < 48)
        {
            values[i + 1] = 0b01101101;
        }
        if (i + 2 < 48)
        {
            values[i + 2] = 0b11011011;
        }
    }

    // Only 11s
    for (size_t i = 48; i < 112; i++)
    {
        values[i] = 0b10111011;
    }

    size_t outputSize = 256;

    auto result = FixedLength::cpuDecompress(outputSize, bits, bitsCount, values, valuesCount);

    uint8_t expectedData[256];
    for (size_t i = 0; i < 128; i++)
    {
        expectedData[i] = 6;
    }
    for (size_t i = 128; i < 256; i++)
    {
        expectedData[i] = 11;
    }

    TEST_CHECK_(result.size == outputSize, "%zu is equal to %zu", result.size, outputSize);
    TEST_ARRAYS_EQUAL(expectedData, result.data, outputSize, "%hhu");
}

void test_fl_cpu_decompression_last_frame_not_full(void)
{
    size_t bitsCount = 2;
    size_t valuesCount = 49; // (128 * 3 + 5) / 8
    uint8_t bits[] = {3, 5};
    uint8_t values[49];

    // Only 6s
    for (size_t i = 0; i < 48; i += 3)
    {
        values[i] = 0b10110110;
        if (i + 1 < 48)
        {
            values[i + 1] = 0b01101101;
        }
        if (i + 2 < 48)
        {
            values[i + 2] = 0b11011011;
        }
    }
    // 16 at the end
    values[48] = 0b00010000;

    size_t outputSize = 129;

    auto result = FixedLength::cpuDecompress(outputSize, bits, bitsCount, values, valuesCount);

    uint8_t expectedData[129];
    for (size_t i = 0; i < 128; i++)
    {
        expectedData[i] = 6;
    }
    expectedData[128] = 16;

    TEST_CHECK_(result.size == outputSize, "%zu is equal to %zu", result.size, outputSize);
    TEST_ARRAYS_EQUAL(expectedData, result.data, outputSize, "%hhu");
}

// Compression + Decompression

void test_fl_cpu_compression_decompression_simple_example(void)
{
    uint8_t data[] = {1, 2, 3, 4, 5, 6, 7};
    size_t dataSize = 7;

    auto compressed = FixedLength::cpuCompress(data, dataSize);
    auto decompressed = FixedLength::cpuDecompress(compressed.inputSize, compressed.outputBits, compressed.bitsSize, compressed.outputValues, compressed.valuesSize);
    TEST_CHECK_(decompressed.size == dataSize, "%zu is equal to %zu", decompressed.size, dataSize);
    TEST_ARRAYS_EQUAL(data, decompressed.data, dataSize, "%hhu");
}

void test_fl_cpu_compression_decompression_all_zeroes(void)
{
    uint8_t data[] = {0, 0, 0, 0, 0, 0, 0, 0};
    size_t dataSize = 8;

    auto compressed = FixedLength::cpuCompress(data, dataSize);
    auto decompressed = FixedLength::cpuDecompress(compressed.inputSize, compressed.outputBits, compressed.bitsSize, compressed.outputValues, compressed.valuesSize);
    TEST_CHECK_(decompressed.size == dataSize, "%zu is equal to %zu", decompressed.size, dataSize);
    TEST_ARRAYS_EQUAL(data, decompressed.data, dataSize, "%hhu");
}

void test_fl_cpu_compression_decompression_all_ones(void)
{
    uint8_t data[] = {255, 255, 255, 255};
    size_t dataSize = 4;

    auto compressed = FixedLength::cpuCompress(data, dataSize);
    auto decompressed = FixedLength::cpuDecompress(compressed.inputSize, compressed.outputBits, compressed.bitsSize, compressed.outputValues, compressed.valuesSize);
    TEST_CHECK_(decompressed.size == dataSize, "%zu is equal to %zu", decompressed.size, dataSize);
    TEST_ARRAYS_EQUAL(data, decompressed.data, dataSize, "%hhu");
}

void test_fl_cpu_compression_decompression_mixed_values(void)
{
    uint8_t data[] = {128, 64, 32, 16, 8, 4, 2, 1};
    size_t dataSize = 8;

    auto compressed = FixedLength::cpuCompress(data, dataSize);
    auto decompressed = FixedLength::cpuDecompress(compressed.inputSize, compressed.outputBits, compressed.bitsSize, compressed.outputValues, compressed.valuesSize);
    TEST_CHECK_(decompressed.size == dataSize, "%zu is equal to %zu", decompressed.size, dataSize);
    TEST_ARRAYS_EQUAL(data, decompressed.data, dataSize, "%hhu");
}

void test_fl_cpu_compression_decompression_single_value(void)
{
    uint8_t data[] = {42};
    size_t dataSize = 1;

    auto compressed = FixedLength::cpuCompress(data, dataSize);
    auto decompressed = FixedLength::cpuDecompress(compressed.inputSize, compressed.outputBits, compressed.bitsSize, compressed.outputValues, compressed.valuesSize);
    TEST_CHECK_(decompressed.size == dataSize, "%zu is equal to %zu", decompressed.size, dataSize);
    TEST_ARRAYS_EQUAL(data, decompressed.data, dataSize, "%hhu");
}

void test_fl_cpu_compression_decompression_two_frames(void)
{
    uint8_t data[130];
    for (size_t i = 0; i < 130; i++)
    {
        data[i] = i % 4;
    }
    size_t dataSize = 130;

    auto compressed = FixedLength::cpuCompress(data, dataSize);
    auto decompressed = FixedLength::cpuDecompress(compressed.inputSize, compressed.outputBits, compressed.bitsSize, compressed.outputValues, compressed.valuesSize);
    TEST_CHECK_(decompressed.size == dataSize, "%zu is equal to %zu", decompressed.size, dataSize);
    TEST_ARRAYS_EQUAL(data, decompressed.data, dataSize, "%hhu");
}

void test_fl_cpu_compression_decompression_three_frames(void)
{
    uint8_t data[384]; // 3 * 128
    for (size_t i = 0; i < 384; i++)
    {
        data[i] = i % 16; // Values 0-15 repeating
    }
    size_t dataSize = 384;

    auto compressed = FixedLength::cpuCompress(data, dataSize);
    auto decompressed = FixedLength::cpuDecompress(compressed.inputSize, compressed.outputBits, compressed.bitsSize, compressed.outputValues, compressed.valuesSize);
    TEST_CHECK_(decompressed.size == dataSize, "%zu is equal to %zu", decompressed.size, dataSize);
    TEST_ARRAYS_EQUAL(data, decompressed.data, dataSize, "%hhu");
}

void test_fl_cpu_compression_decompression_partial_last_frame(void)
{
    uint8_t data[250];
    for (size_t i = 0; i < 250; i++)
    {
        data[i] = 1;
    }
    size_t dataSize = 250;

    auto compressed = FixedLength::cpuCompress(data, dataSize);
    auto decompressed = FixedLength::cpuDecompress(compressed.inputSize, compressed.outputBits, compressed.bitsSize, compressed.outputValues, compressed.valuesSize);
    TEST_CHECK_(decompressed.size == dataSize, "%zu is equal to %zu", decompressed.size, dataSize);
    TEST_ARRAYS_EQUAL(data, decompressed.data, dataSize, "%hhu");
}

void test_fl_cpu_compression_decompression_different_bits_per_frame(void)
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
    size_t dataSize = 256;

    auto compressed = FixedLength::cpuCompress(data, dataSize);
    auto decompressed = FixedLength::cpuDecompress(compressed.inputSize, compressed.outputBits, compressed.bitsSize, compressed.outputValues, compressed.valuesSize);
    TEST_CHECK_(decompressed.size == dataSize, "%zu is equal to %zu", decompressed.size, dataSize);
    TEST_ARRAYS_EQUAL(data, decompressed.data, dataSize, "%hhu");
}

void test_fl_cpu_compression_decompression_random(void)
{
    uint8_t data[10000];
    for (size_t i = 0; i < 10000; i++)
    {
        data[i] = rand() % 256; // 110
    }
    size_t dataSize = 10000;

    auto compressed = FixedLength::cpuCompress(data, dataSize);
    auto decompressed = FixedLength::cpuDecompress(compressed.inputSize, compressed.outputBits, compressed.bitsSize, compressed.outputValues, compressed.valuesSize);
    TEST_CHECK_(decompressed.size == dataSize, "%zu is equal to %zu", decompressed.size, dataSize);
    TEST_ARRAYS_EQUAL(data, decompressed.data, dataSize, "%hhu");
}

TEST_LIST = {
    // Compressions
    {"test_fl_cpu_compression_simple_example", test_fl_cpu_compression_simple_example},
    {"test_fl_cpu_compression_all_zeroes", test_fl_cpu_compression_all_zeroes},
    {"test_fl_cpu_compression_all_ones", test_fl_cpu_compression_all_ones},
    {"test_fl_cpu_compression_mixed_values", test_fl_cpu_compression_mixed_values},
    {"test_fl_cpu_compression_single_value", test_fl_cpu_compression_single_value},
    {"test_fl_cpu_compression_two_frames", test_fl_cpu_compression_two_frames},
    {"test_fl_cpu_compression_three_frames", test_fl_cpu_compression_three_frames},
    {"test_fl_cpu_compression_partial_last_frame", test_fl_cpu_compression_partial_last_frame},
    {"test_fl_cpu_compression_different_bits_per_frame", test_fl_cpu_compression_different_bits_per_frame},
    // Decompressions
    {"test_fl_cpu_decompression_simple_example", test_fl_cpu_decompression_simple_example},
    {"test_fl_cpu_decompression_single_element", test_fl_cpu_decompression_single_element},
    {"test_fl_cpu_decompression_single_frame_partial", test_fl_cpu_decompression_single_frame_partial},
    {"test_fl_cpu_decompression_two_frames", test_fl_cpu_decompression_two_frames},
    {"test_fl_cpu_decompression_last_frame_not_full", test_fl_cpu_decompression_last_frame_not_full},
    // Compression + Decompression
    {"test_fl_cpu_compression_decompression_simple_example", test_fl_cpu_compression_decompression_simple_example},
    {"test_fl_cpu_compression_decompression_all_zeroes", test_fl_cpu_compression_decompression_all_zeroes},
    {"test_fl_cpu_compression_decompression_all_ones", test_fl_cpu_compression_decompression_all_ones},
    {"test_fl_cpu_compression_decompression_mixed_values", test_fl_cpu_compression_decompression_mixed_values},
    {"test_fl_cpu_compression_decompression_single_value", test_fl_cpu_compression_decompression_single_value},
    {"test_fl_cpu_compression_decompression_two_frames", test_fl_cpu_compression_decompression_two_frames},
    {"test_fl_cpu_compression_decompression_three_frames", test_fl_cpu_compression_decompression_three_frames},
    {"test_fl_cpu_compression_decompression_partial_last_frame", test_fl_cpu_compression_decompression_partial_last_frame},
    {"test_fl_cpu_compression_decompression_different_bits_per_frame", test_fl_cpu_compression_decompression_different_bits_per_frame},
    {"test_fl_cpu_compression_decompression_random", test_fl_cpu_compression_decompression_random},
    {nullptr, nullptr}};