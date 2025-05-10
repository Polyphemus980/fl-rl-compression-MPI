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

    MpiData initMPI();

    FLCompressed gpuMPICompress(uint8_t *data, size_t size, MpiData mpiData);

    int SendFLCompressed(const FLCompressed &data, int destination, int tag, MPI_Comm comm);
    FLCompressed ReceiveFLCompressed(int source, int tag, MPI_Comm comm, MPI_Status *status);
    FLCompressed MergeFLCompressed(const FLCompressed *structs, int count);
}

#endif