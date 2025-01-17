#ifndef UTILS_H
#define UTILS_H

#include <stdexcept>

#define CHECK_CUDA(call)                                                      \
    {                                                                         \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess)                                               \
        {                                                                     \
            throw std::runtime_error(std::string("CUDA Error: ") +            \
                                     cudaGetErrorString(err) +                \
                                     " in file " + __FILE__ +                 \
                                     " at line " + std::to_string(__LINE__)); \
        }                                                                     \
    }

#endif // UTILS_H