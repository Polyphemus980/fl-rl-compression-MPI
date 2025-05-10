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
        
        // Prepare local metadata
        MetaData localMeta = {
            compressedData.bitsSize,
            compressedData.valuesSize,
            compressedData.inputSize
        };
        
        // Gather metadata from all ranks to rank 0
        std::vector<MetaData> allMeta;
        if (rank == 0) {
            allMeta.resize(nodesCount);
        }
        
        cpuTimer.start();
        MPI_Gather(&localMeta, sizeof(MetaData), MPI_BYTE,
                rank == 0 ? allMeta.data() : nullptr, 
                sizeof(MetaData), MPI_BYTE,
                0, MPI_COMM_WORLD);
        
        // On rank 0, allocate arrays for gathering compressed data
        std::vector<int> bitsRecvCounts, bitsDisplacements;
        std::vector<int> valuesRecvCounts, valuesDisplacements;
        std::vector<uint8_t> allBits, allValues;
        size_t totalBitsSize = 0, totalValuesSize = 0;
        
        if (rank == 0) {
            bitsRecvCounts.resize(nodesCount);
            bitsDisplacements.resize(nodesCount);
            valuesRecvCounts.resize(nodesCount);
            valuesDisplacements.resize(nodesCount);
            
            // Calculate displacements and total sizes
            for (int i = 0; i < nodesCount; i++) {
                bitsRecvCounts[i] = static_cast<int>(allMeta[i].bitsSize);
                bitsDisplacements[i] = static_cast<int>(totalBitsSize);
                totalBitsSize += allMeta[i].bitsSize;
                
                valuesRecvCounts[i] = static_cast<int>(allMeta[i].valuesSize);
                valuesDisplacements[i] = static_cast<int>(totalValuesSize);
                totalValuesSize += allMeta[i].valuesSize;
            }
            
            // Allocate space for received data
            allBits.resize(totalBitsSize);
            allValues.resize(totalValuesSize);
        }
            
        // Gather bits data
        MPI_Gatherv(
            compressedData.outputBits, static_cast<int>(compressedData.bitsSize), MPI_UNSIGNED_CHAR,
            rank == 0 ? allBits.data() : nullptr, 
            rank == 0 ? bitsRecvCounts.data() : nullptr,
            rank == 0 ? bitsDisplacements.data() : nullptr, 
            MPI_UNSIGNED_CHAR,
            0, MPI_COMM_WORLD
        );
        
        // Gather values data
        MPI_Gatherv(
            compressedData.outputValues, static_cast<int>(compressedData.valuesSize), MPI_UNSIGNED_CHAR,
            rank == 0 ? allValues.data() : nullptr, 
            rank == 0 ? valuesRecvCounts.data() : nullptr,
            rank == 0 ? valuesDisplacements.data() : nullptr, 
            MPI_UNSIGNED_CHAR,
            0, MPI_COMM_WORLD
        );
        
        cpuTimer.end();
        if (rank == 0) {
            cpuTimer.printResult("Gather compressed data from all nodes");
            
            // Construct FLCompressed objects for each rank
            FLCompressed *compressedWholeData = new FLCompressed[nodesCount];
            
            for (int i = 0; i < nodesCount; i++) {
                uint8_t *nodeBits = nullptr;
                uint8_t *nodeValues = nullptr;
                
                if (allMeta[i].bitsSize > 0) {
                    nodeBits = new uint8_t[allMeta[i].bitsSize];
                    std::memcpy(nodeBits, allBits.data() + bitsDisplacements[i], allMeta[i].bitsSize);
                }
                
                if (allMeta[i].valuesSize > 0) {
                    nodeValues = new uint8_t[allMeta[i].valuesSize];
                    std::memcpy(nodeValues, allValues.data() + valuesDisplacements[i], allMeta[i].valuesSize);
                }
                
                compressedWholeData[i] = FLCompressed(
                    nodeBits, allMeta[i].bitsSize,
                    nodeValues, allMeta[i].valuesSize,
                    allMeta[i].inputSize
                );
            }
            
            // Set our compressed data for rank 0 (avoiding memory allocation/copy overhead)
            if (compressedData.bitsSize > 0 || compressedData.valuesSize > 0) {
                // Keep our original allocation for rank 0 data (it will be managed by MergeFLCompressed)
                compressedWholeData[0] = std::move(compressedData);
            }
            
            MPI_Finalize();
            // Merge all the compressed data and return
            return FixedLength::FLCompressed::MergeFLCompressed(compressedWholeData, nodesCount);
        }
        else {
            cpuTimer.printResult("Send compressed data to node 0");
            MPI_Finalize();
            exit(0);
            // We need to return something here even though this path calls exit()
            // This is to satisfy the compiler
            return FLCompressed(); 
        }
    }
}