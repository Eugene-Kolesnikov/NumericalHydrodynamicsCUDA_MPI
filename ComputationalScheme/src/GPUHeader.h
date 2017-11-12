#ifndef GPUHEADER_H
#define GPUHEADER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <exception>

#define HANDLE_CUERROR(call) {                                        \
    cudaError err = call;                                             \
    if(err != cudaSuccess) {                                          \
        std::string error = std::string("CUDA error in file '") +     \
            std::string(__FILE__) + std::string("' in line ") +       \
            std::to_string(__LINE__) + std::string(": ") +            \
            std::string(cudaGetErrorString(err));                  \
        throw std::runtime_error(error);                              \
    }                                                                 \
} while (0)

#endif /* GPUHEADER_H */