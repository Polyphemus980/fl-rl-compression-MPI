#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "./timers/cpu_timer.cuh"
#include "file_io.cuh"

namespace FileIO
{
    FileData::FileData()
    {
        data = nullptr;
        size = 0;
    }

    FileData::FileData(uint8_t *data, size_t size)
    {
        this->data = data;
        this->size = size;
    }

    FileData::FileData(FixedLength::FLDecompressed flDecompressed)
    {
        data = flDecompressed.data;
        size = flDecompressed.size;
    }

    FileData loadFileMpi(const char* path, MpiData mpiData) {
        Timers::CpuTimer cpuTimer;
        cpuTimer.start();
    
        // Open file
        FILE* file = fopen(path, "rb");
        if (file == nullptr) {
            throw std::runtime_error("[FileIO] Cannot open file");
        }
    
        // Get file size
        fseek(file, 0, SEEK_END);
        size_t fileSize = ftell(file);
        fseek(file, 0, SEEK_SET);
    
        // Round up total file size to multiple of 128
        constexpr size_t ALIGNMENT = 128;
        size_t alignedFileSize = (fileSize + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    
        // Divide work across nodes
        size_t alignedPerNode = alignedFileSize / mpiData.nodesCount;
    
        // For all nodes except last, they read exactly alignedPerNode
        // Last node reads remaining bytes (might go slightly past original fileSize)
        size_t nodeOffset = mpiData.rank * alignedPerNode;
        size_t nodeReadSize = (mpiData.rank == mpiData.nodesCount - 1)
            ? fileSize - nodeOffset
            : alignedPerNode;
    
        if (nodeOffset >= fileSize) {
            nodeReadSize = 0;  // Nothing to read
        }
    
        // Allocate buffer and read
        uint8_t* fileData = reinterpret_cast<uint8_t*>(malloc(nodeReadSize));
        if (fileData == nullptr) {
            fclose(file);
            throw std::runtime_error("[FileIO] Cannot allocate memory");
        }
    
        fseek(file, nodeOffset, SEEK_SET);
        size_t readCount = fread(fileData, sizeof(uint8_t), nodeReadSize, file);
        if (readCount != nodeReadSize) {
            free(fileData);
            fclose(file);
            throw std::runtime_error("[FileIO] Cannot read file content");
        }
    
        fclose(file);
        cpuTimer.end();
        cpuTimer.printResult("Load data from file");
    
        return FileData(fileData, nodeReadSize);
    }
        
    FileData loadFile(const char *path)
    {
        Timers::CpuTimer cpuTimer;

        cpuTimer.start();

        // Open file
        FILE *file = fopen(path, "rb");
        if (file == nullptr)
        {
            throw std::runtime_error("[FileIO] Cannot open file");
        }

        // Check file size
        fseek(file, 0, SEEK_END);
        size_t fileSize = ftell(file);
        fseek(file, 0, SEEK_SET);

        // Allocate memory for file data
        uint8_t *fileData = reinterpret_cast<uint8_t *>(malloc(sizeof(uint8_t) * fileSize));
        if (fileData == nullptr)
        {
            fclose(file);
            throw std::runtime_error("[FileIO] Cannot allocate memory");
        }

        // Read file data
        size_t readCount = fread(fileData, sizeof(uint8_t), fileSize, file);
        if (readCount != fileSize)
        {
            free(fileData);
            fclose(file);
            throw std::runtime_error("[FileIO] Cannot read file content");
        }

        // Cleanuo
        fclose(file);

        cpuTimer.end();
        cpuTimer.printResult("Load data from file");

        return FileData(fileData, fileSize);
    }

    FixedLength::FLCompressed loadCompressedFL(const char *path)
    {
        Timers::CpuTimer cpuTimer;

        cpuTimer.start();

        // Open file
        FILE *file = fopen(path, "rb");
        if (file == nullptr)
        {
            throw std::runtime_error("[FileIO] Cannot open file");
        }

        // Read input size
        size_t inputSize;
        size_t readCount = fread(&inputSize, sizeof(size_t), 1, file);
        if (readCount != 1)
        {
            fclose(file);
            throw std::runtime_error("[FileIO] Cannot read file content");
        }

        // Read bits size
        size_t bitsSize;
        readCount = fread(&bitsSize, sizeof(size_t), 1, file);
        if (readCount != 1)
        {
            fclose(file);
            throw std::runtime_error("[FileIO] Cannot read file content");
        }

        // Read values size
        size_t valuesSize;
        readCount = fread(&valuesSize, sizeof(size_t), 1, file);
        if (readCount != 1)
        {
            fclose(file);
            throw std::runtime_error("[FileIO] Cannot read file content");
        }

        // Read bits array
        uint8_t *bits = reinterpret_cast<uint8_t *>(malloc(sizeof(uint8_t) * bitsSize));
        if (bits == nullptr)
        {
            fclose(file);
            throw std::runtime_error("Cannot allocate memory");
        }
        readCount = fread(bits, sizeof(uint8_t), bitsSize, file);
        if (readCount != bitsSize)
        {
            fclose(file);
            free(bits);
            throw std::runtime_error("[FileIO] Cannot read file content");
        }

        // Read values array
        uint8_t *values = reinterpret_cast<uint8_t *>(malloc(sizeof(uint8_t) * valuesSize));
        if (values == nullptr)
        {
            fclose(file);
            throw std::runtime_error("Cannot allocate memory");
        }
        readCount = fread(values, sizeof(uint8_t), valuesSize, file);
        if (readCount != valuesSize)
        {
            fclose(file);
            free(bits);
            free(values);
            throw std::runtime_error("[FileIO] Cannot read file content");
        }

        cpuTimer.end();
        cpuTimer.printResult("Load data from file");

        return FixedLength::FLCompressed(bits, bitsSize, values, valuesSize, inputSize);
    }

    void saveFile(const char *path, FileData fileData)
    {
        Timers::CpuTimer cpuTimer;

        cpuTimer.start();

        // Open file
        FILE *file = fopen(path, "wb");
        if (file == nullptr)
        {
            throw std::runtime_error("[FileIO] Cannot open file");
        }

        // Save content to file
        size_t writeCount = fwrite(fileData.data, sizeof(uint8_t), fileData.size, file);
        if (writeCount != fileData.size)
        {
            fclose(file);
            throw std::runtime_error("[FileIO] Cannot write to file");
        }

        // Cleanup
        fclose(file);

        cpuTimer.end();
        cpuTimer.printResult("Save data to file");
    }

    void saveCompressedFL(const char *path, FixedLength::FLCompressed flCompressed)
    {
        Timers::CpuTimer cpuTimer;

        cpuTimer.start();

        // Open file
        FILE *file = fopen(path, "wb");
        if (file == nullptr)
        {
            throw std::runtime_error("[FileIO] Cannot open file");
        }

        // Save input size
        size_t writeCount = fwrite(&flCompressed.inputSize, sizeof(size_t), 1, file);
        if (writeCount != 1)
        {
            fclose(file);
            throw std::runtime_error("[FileIO] Cannot write to file");
        }

        // Save bits size
        writeCount = fwrite(&flCompressed.bitsSize, sizeof(size_t), 1, file);
        if (writeCount != 1)
        {
            fclose(file);
            throw std::runtime_error("[FileIO] Cannot write to file");
        }

        // Save values size
        writeCount = fwrite(&flCompressed.valuesSize, sizeof(size_t), 1, file);
        if (writeCount != 1)
        {
            fclose(file);
            throw std::runtime_error("[FileIO] Cannot write to file");
        }

        // Save bits array
        writeCount = fwrite(flCompressed.outputBits, sizeof(uint8_t), flCompressed.bitsSize, file);
        if (writeCount != flCompressed.bitsSize)
        {
            fclose(file);
            throw std::runtime_error("[FileIO] Cannot write to file");
        }

        // Save values array
        writeCount = fwrite(flCompressed.outputValues, sizeof(uint8_t), flCompressed.valuesSize, file);
        if (writeCount != flCompressed.valuesSize)
        {
            fclose(file);
            throw std::runtime_error("[FileIO] Cannot write to file");
        }

        // Cleanup
        fclose(file);

        cpuTimer.end();
        cpuTimer.printResult("Save data to file");
    }
} // FileIO