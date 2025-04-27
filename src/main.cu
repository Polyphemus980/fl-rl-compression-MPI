#include <iostream>
#include <cstdio>
#include <cstdint>
#include <stdexcept>

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
    FileIO::FileData content;
    try
    {
        content = FileIO::loadFile(input);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR]: " << e.what() << '\n';
        return;
    }

    if (method == ArgsParser::Method::FixedLength || method == ArgsParser::Method::FixedLengthCPU)
    {
        // Fixed length
        FixedLength::FLCompressed compressed;
        try
        {
            if (method == ArgsParser::Method::FixedLength)
            {
                compressed = FixedLength::gpuMPICompress(content.data, content.size);
            }
            else
            {
                compressed = FixedLength::cpuCompress(content.data, content.size);
            }
            FileIO::saveCompressedFL(output, compressed);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ERROR]: " << e.what() << '\n';
        }
        free(compressed.outputValues);
        free(compressed.outputBits);
    }
    else
    {
        // Run length
        RunLength::RLCompressed compressed;
        try
        {
            if (method == ArgsParser::Method::RunLength)
            {
                compressed = RunLength::gpuCompress(content.data, content.size);
            }
            else
            {
                compressed = RunLength::cpuCompress(content.data, content.size);
            }
            FileIO::saveCompressedRL(output, compressed);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ERROR]: " << e.what() << '\n';
        }
        free(compressed.outputValues);
        free(compressed.outputCounts);
    }
    free(content.data);
}

void decompress(ArgsParser::Method method, const char *input, const char *output)
{
    FileIO::FileData fd;
    bool canSaveFile = true;
    if (method == ArgsParser::Method::FixedLength || method == ArgsParser::Method::FixedLengthCPU)
    {
        // Fixed length
        FixedLength::FLCompressed compressed;
        try
        {
            compressed = FileIO::loadCompressedFL(input);
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
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ERROR]: " << e.what() << '\n';
            canSaveFile = false;
        }
        free(compressed.outputValues);
        free(compressed.outputBits);
    }
    else
    {
        // Run length
        RunLength::RLCompressed compressed;
        try
        {
            compressed = FileIO::loadCompressedRL(input);
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
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ERROR]: " << e.what() << '\n';
            canSaveFile = false;
        }
        free(compressed.outputValues);
        free(compressed.outputCounts);
    }
    if (canSaveFile)
    {
        try
        {
            FileIO::saveFile(output, fd);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ERROR]: " << e.what() << '\n';
        }
    }
    free(fd.data);
}
