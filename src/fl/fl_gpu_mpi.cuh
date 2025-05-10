#ifndef FL_GPU_MPI_H
#define FL_GPU_MPI_H

#include <mpi.h>

#include "fl_common.cuh"

namespace FixedLength {
    struct MpiData
    {
        int rank;
        int nodesCount;

        MpiData() : rank(0), nodesCount(0) {}

        MpiData(int rank, int nodesCount)
            : rank(rank), nodesCount(nodesCount) {}
    };

    struct MetaData {
        size_t bitsSize;
        size_t valuesSize;
        size_t inputSize;
    };

    MpiData initMPI();

    FLCompressed gpuMPICompress(uint8_t *data, size_t size, MpiData mpiData);
}

#endif