#pragma once

#include "nccl.h"
struct MpiData
{
    int rank;
    int nodesCount;

    MpiData() : rank(0), nodesCount(0) {}

    MpiData(int rank, int nodesCount)
        : rank(rank), nodesCount(nodesCount) {}
};

struct MpiNcclData
{
    int rank;
    int nodesCount;
    int device;
    ncclComm_t comm;

    MpiNcclData() : rank(0), nodesCount(0), device(0), comm(nullptr) {}
    MpiNcclData(int r, int n, int d, ncclComm_t c) : rank(r), nodesCount(n), device(d), comm(c) {}
};