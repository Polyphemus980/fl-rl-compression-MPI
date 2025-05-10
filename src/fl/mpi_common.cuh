#pragma once

struct MpiData
{
    int rank;
    int nodesCount;

    MpiData() : rank(0), nodesCount(0) {}

    MpiData(int rank, int nodesCount)
        : rank(rank), nodesCount(nodesCount) {}
};