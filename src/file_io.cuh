#ifndef FILE_IO
#define FILE_IO

#include <cstdint>
#include "./fl/fl_common.cuh"
#include "./rl/rl_common.cuh"

namespace FileIO
{
    struct FileData
    {
        uint8_t *data;
        size_t size;

        FileData();
        FileData(uint8_t *data, size_t size);
        FileData(FixedLength::FLDecompressed flDecompressed);
        FileData(RunLength::RLDecompressed rlDecompressed);
    };

    FileData loadFile(const char *path);
    FixedLength::FLCompressed loadCompressedFL(const char *path);
    RunLength::RLCompressed loadCompressedRL(const char *path);

    void saveFile(const char *path, FileData fileData);
    void saveCompressedFL(const char *path, FixedLength::FLCompressed flCompressed);
    void saveCompressedRL(const char *path, RunLength::RLCompressed rlCompressed);

} // FileIO

#endif // FILE_IO