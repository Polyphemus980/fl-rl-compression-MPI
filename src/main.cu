#include <iostream>
#include <cstdio>
#include <cstdint>
#include <chrono>

#include "args_parser.cuh"
#include "file_io.cuh"
#include "./fl/fl_cpu.cuh"
#include "./fl/fl_gpu.cuh"
#include "./rl/rl_cpu.cuh"
#include "./rl/rl_gpu.cuh"

void compress(ArgsParser::Method method, const char *input, const char *output);
void decompress(ArgsParser::Method method, const char *input, const char *output);

int main(int argc, char **argv)
{
    auto args = ArgsParser::parseArguments(argc, argv);

    switch (args.operation)
    {
    case ArgsParser::Operation::Compression:
        compress(args.method, args.inputFile, args.outputFile);
        break;
    case ArgsParser::Operation::Decompression:
        decompress(args.method, args.inputFile, args.outputFile);
        break;
    }

    return 0;
}

void compress(ArgsParser::Method method, const char *input, const char *output)
{
    auto content = FileIO::loadFile(input);
    if (method == ArgsParser::Method::FixedLength || method == ArgsParser::Method::FixedLengthCPU)
    {
        // Fixed length
        FixedLength::FLCompressed compressed;
        if (method == ArgsParser::Method::FixedLength)
        {
            compressed = FixedLength::gpuCompress(content.data, content.size);
        }
        else
        {
            compressed = FixedLength::cpuCompress(content.data, content.size);
        }
        FileIO::saveCompressedFL(output, compressed);
        free(compressed.outputValues);
        free(compressed.outputBits);
    }
    else
    {
        // Run length
        RunLength::RLCompressed compressed;
        if (method == ArgsParser::Method::RunLength)
        {
            compressed = RunLength::gpuCompress(content.data, content.size);
        }
        else
        {
            compressed = RunLength::cpuCompress(content.data, content.size);
        }
        FileIO::saveCompressedRL(output, compressed);
        free(compressed.outputValues);
        free(compressed.outputCounts);
    }
    free(content.data);
}

void decompress(ArgsParser::Method method, const char *input, const char *output)
{
    FileIO::FileData fd;
    if (method == ArgsParser::Method::FixedLength || method == ArgsParser::Method::FixedLengthCPU)
    {
        // Fixed length
        auto compressed = FileIO::loadCompressedFL(input);
        if (method == ArgsParser::Method::FixedLength)
        {
            auto decompressed = FixedLength::gpuDecompress(compressed.inputSize, compressed.outputBits, compressed.bitsSize, compressed.outputValues, compressed.valuesSize);
            fd = FileIO::FileData(decompressed);
        }
        else
        {
            auto decompressed = FixedLength::cpuDecompress(compressed.inputSize, compressed.outputBits, compressed.bitsSize, compressed.outputValues, compressed.valuesSize);
            fd = FileIO::FileData(decompressed);
        }
        free(compressed.outputValues);
        free(compressed.outputBits);
    }
    else
    {
        // Run length
        auto compressed = FileIO::loadCompressedRL(input);
        if (method == ArgsParser::Method::RunLength)
        {
            auto decompressed = RunLength::gpuDecompress(compressed.outputValues, compressed.outputCounts, compressed.count);
            fd = FileIO::FileData(decompressed);
        }
        else
        {
            auto decompressed = RunLength::cpuDecompress(compressed.outputValues, compressed.outputCounts, compressed.count);
            fd = FileIO::FileData(decompressed);
        }
        free(compressed.outputValues);
        free(compressed.outputCounts);
    }
    FileIO::saveFile(output, fd);
    free(fd.data);
}
