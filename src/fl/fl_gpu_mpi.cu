#include <mpi.h>

#include "fl_gpu.cuh"
#include "fl_gpu_mpi.cuh"
#include "fl_common.cuh"
#include "../utils.cuh"
#include "../timers/cpu_timer.cuh"
#include "../timers/gpu_timer.cuh"

namespace FixedLength {
    MpiData initMPI()
    {
        int rank, nodesCount;
        MPI_Init(NULL, NULL);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nodesCount);
        
        printf("[INFO] Process %d of %d started\n", rank, nodesCount);
        return MpiData(rank, nodesCount);
    }

    FLCompressed gpuMPICompress(uint8_t *data, size_t size, MpiData mpiData)
    {
        Timers::CpuTimer cpuTimer;
        
        int rank = mpiData.rank;
        int nodesCount = mpiData.nodesCount;
        FLCompressed compressedData = gpuCompress(data, size);
        
        if (rank == 0)
        {
            FLCompressed *compressedWholeData = new FLCompressed[nodesCount];
            compressedWholeData[rank] = compressedData;
            cpuTimer.start();
            for (int i = 1; i < nodesCount; i++)
            {
                compressedWholeData[i] = FixedLength::FLCompressed::ReceiveFLCompressed(i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            cpuTimer.end();
            cpuTimer.printResult("Receive compressed data from all nodes");
            MPI_Finalize();
            return FixedLength::FLCompressed::MergeFLCompressed(compressedWholeData, nodesCount);
        }
        else
        {
            cpuTimer.start();
            FixedLength::FLCompressed::SendFLCompressed(compressedData, 0, 0, MPI_COMM_WORLD);
            cpuTimer.end();
            cpuTimer.printResult("Send compressed data to node 0");
            MPI_Finalize();
            exit(0);
        }
    }
}