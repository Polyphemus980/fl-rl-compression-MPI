#ifndef FL_GPU_MPI_H
#define FL_GPU_MPI_H

namespace FixedLength {
    struct MpiData
    {
        int rank;
        int nodesCount;

        MpiData() : rank(0), nodesCount(0) {}

        MpiData(int rank, int nodesCount)
            : rank(rank), nodesCount(nodesCount) {}
    };
}

#endif