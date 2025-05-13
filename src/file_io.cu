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
    
        // Compute basic chunk sizes
        size_t baseChunk = fileSize / mpiData.nodesCount;
        size_t remainder = fileSize % mpiData.nodesCount;
    
        // Calculate actual start and size for this node
        size_t nodeStart = mpiData.rank * baseChunk + std::min(static_cast<size_t>(mpiData.rank), remainder);
        size_t nodeSize = baseChunk + (mpiData.rank < remainder ? 1 : 0);
    
        // Round up to 128-byte alignment (for processing purposes)
        constexpr size_t ALIGNMENT = 128;
        size_t alignedNodeSize = (nodeSize + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    
        // Allocate aligned buffer and zero out padding area
        uint8_t* fileData = reinterpret_cast<uint8_t*>(malloc(alignedNodeSize));
        if (fileData == nullptr) {
            fclose(file);
            throw std::runtime_error("[FileIO] Cannot allocate memory");
        }
        memset(fileData, 0, alignedNodeSize);  // Zero padding
    
        // Read only the real file content
        fseek(file, nodeStart, SEEK_SET);
        size_t readCount = fread(fileData, sizeof(uint8_t), nodeSize, file);
        if (readCount != nodeSize) {
            free(fileData);
            fclose(file);
            throw std::runtime_error("[FileIO] Cannot read file content");
        }
    
        fclose(file);
        cpuTimer.end();
        cpuTimer.printResult("Load data from file");
    
        return FileData(fileData, alignedNodeSize);
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