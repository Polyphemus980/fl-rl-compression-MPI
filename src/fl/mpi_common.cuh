#pragma once

struct MpiData
{
    int rank;
    int nodesCount;

    MpiData(int rank, int nodesCount)
        : rank(rank), nodesCount(nodesCount) {}
};