#ifndef FL_COMMON_H
#define FL_COMMON_H

#include <cstdint>
#include <mpi.h>
namespace FixedLength
{
    // Number of bytes per frame
    static constexpr size_t FRAME_LENGTH = 128;

    struct FLCompressed
    {
        uint8_t *outputBits;
        size_t bitsSize;
        uint8_t *outputValues;
        size_t valuesSize;
        size_t inputSize;

        FLCompressed()
        {
            this->outputBits = nullptr;
            this->bitsSize = 0;
            this->outputValues = nullptr;
            this->valuesSize = 0;
            this->inputSize = 0;
        };
        FLCompressed(uint8_t *outputBits, size_t bitsSize, uint8_t *outputValues, size_t valuesSize, size_t inputSize)
        {
            this->outputBits = outputBits;
            this->bitsSize = bitsSize;
            this->outputValues = outputValues;
            this->valuesSize = valuesSize;
            this->inputSize = inputSize;
        }

        static int SendFLCompressed(const FLCompressed &data, int destination, int tag, MPI_Comm comm)
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
        static FLCompressed ReceiveFLCompressed(int source, int tag, MPI_Comm comm, MPI_Status *status)
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

        static FLCompressed MergeFLCompressed(const FLCompressed *structs, int count)
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
    };

    struct FLDecompressed
    {
        uint8_t *data;
        size_t size;

        FLDecompressed()
        {
            this->data = nullptr;
            this->size = 0;
        };

        FLDecompressed(uint8_t *data, size_t size)
        {
            this->data = data;
            this->size = size;
        };
    };

    inline __device__ __host__ uint8_t countLeadingZeroes(uint8_t value)
    {
        if (value == 0)
        {
            return 8;
        }
        uint8_t count = 0;
        uint8_t mask = 1 << 7;
        while (!(value & mask))
        {
            count++;
            value <<= 1;
        }
        return count;
    }
} //  FixedLength

#endif // FL_COMMON_H