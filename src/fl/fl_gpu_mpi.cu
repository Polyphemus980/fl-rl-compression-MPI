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

    int SendFLCompressed(const FLCompressed &data, int destination, int tag, MPI_Comm comm)
    {
        int rank;
        MPI_Comm_rank(comm, &rank);

        // First send the sizes
        size_t sizes[3] = {data.bitsSize, data.valuesSize, data.inputSize};
        MPI_Send(sizes, 3, MPI_UNSIGNED_LONG, destination, tag, comm);

        // Then send the actual data if sizes are non-zero
        if (data.bitsSize > 0 && data.outputBits != nullptr)
        {
            MPI_Send(data.outputBits, data.bitsSize, MPI_UNSIGNED_CHAR, destination, tag + 1, comm);
        }

        if (data.valuesSize > 0 && data.outputValues != nullptr)
        {
            MPI_Send(data.outputValues, data.valuesSize, MPI_UNSIGNED_CHAR, destination, tag + 2, comm);
        }

        return MPI_SUCCESS;
    }

    // To receive the FLCompressed struct
    FLCompressed ReceiveFLCompressed(int source, int tag, MPI_Comm comm, MPI_Status *status)
    {
        // Receive the sizes first
        size_t sizes[3];
        MPI_Recv(sizes, 3, MPI_UNSIGNED_LONG, source, tag, comm, status);


        size_t bitsSize = sizes[0];
        size_t valuesSize = sizes[1];
        size_t inputSize = sizes[2];

        // Allocate memory for the data
        uint8_t *outputBits = nullptr;
        uint8_t *outputValues = nullptr;

        if (bitsSize > 0)
        {
            outputBits = new uint8_t[bitsSize];
            MPI_Recv(outputBits, bitsSize, MPI_UNSIGNED_CHAR, source, tag + 1, comm, status);
        }

        if (valuesSize > 0)
        {
            outputValues = new uint8_t[valuesSize];
            MPI_Recv(outputValues, valuesSize, MPI_UNSIGNED_CHAR, source, tag + 2, comm, status);
        }

        // Create and return the struct
        return FLCompressed(outputBits, bitsSize, outputValues, valuesSize, inputSize);
    }

    FLCompressed MergeFLCompressed(const FLCompressed *structs, int count)
    {
        if (count <= 0)
        {
            return FLCompressed();
        }

        // Calculate total sizes
        size_t totalBitsSize = 0;
        size_t totalValuesSize = 0;
        size_t totalInputSize = 0;

        for (int i = 0; i < count; i++)
        {
            totalBitsSize += structs[i].bitsSize;
            totalValuesSize += structs[i].valuesSize;
            totalInputSize += structs[i].inputSize;
        }

        // Allocate memory for merged data
        uint8_t *mergedBits = nullptr;
        uint8_t *mergedValues = nullptr;

        if (totalBitsSize > 0)
        {
            mergedBits = new uint8_t[totalBitsSize];
        }

        if (totalValuesSize > 0)
        {
            mergedValues = new uint8_t[totalValuesSize];
        }

        // Copy data from each struct
        size_t bitsOffset = 0;
        size_t valuesOffset = 0;

        for (int i = 0; i < count; i++)
        {
            // Copy bits array
            if (structs[i].bitsSize > 0 && structs[i].outputBits != nullptr)
            {
                memcpy(mergedBits + bitsOffset, structs[i].outputBits, structs[i].bitsSize);
                bitsOffset += structs[i].bitsSize;
            }

            // Copy values array
            if (structs[i].valuesSize > 0 && structs[i].outputValues != nullptr)
            {
                memcpy(mergedValues + valuesOffset, structs[i].outputValues, structs[i].valuesSize);
                valuesOffset += structs[i].valuesSize;
            }
        }

        // Create and return the merged struct
        return FLCompressed(mergedBits, totalBitsSize, mergedValues, totalValuesSize, totalInputSize);
    }

}