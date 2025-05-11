#ifndef FILE_IO
#define FILE_IO

#include <cstdint>
#include "./fl/fl_common.cuh"
#include "./fl/mpi_common.cuh"

namespace FileIO
{
    struct FileData
    {
        uint8_t *data;
        size_t size;

        FileData();
        FileData(uint8_t *data, size_t size);
        FileData(FixedLength::FLDecompressed flDecompressed);
    };

    FileData loadFile(const char *path);
    FileData loadFileMpi(const char *path, MpiData mpiData);
    FixedLength::FLCompressed loadCompressedFL(const char *path);
    FileData loadFileNccl(const char *path, MpiNcclData mpiData);
    void saveFile(const char *path, FileData fileData);
    void saveCompressedFL(const char *path, FixedLength::FLCompressed flCompressed);
} // FileIO

#endif // FILE_IO