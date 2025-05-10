#ifndef FL_GPU_MPI_CUH
#define FL_GPU_MPI_CUH

#include <cstdint>
#include <cstddef>
#include "fl_common.cuh"

namespace FixedLength {
    // MPI data structure
    struct MpiData {
        int rank;
        int nodesCount;
        
        MpiData() : rank(0), nodesCount(0) {}
        MpiData(int r, int n) : rank(r), nodesCount(n) {}
    };

    struct MetaData {
        size_t bitsSize;
        size_t valuesSize;
        size_t inputSize;
    };
    // Function declarations
    MpiData initMPI();
    FLCompressed gpuMPICompress(uint8_t *data, size_t size, MpiData mpiData);
    FLCompressed MergeFLCompressed(const FLCompressed *structs, int count);
}

#endif // FL_GPU_MPI_CUH