#ifndef GPUHEADER_H
#define GPUHEADER_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <exception>
#include "../../ComputationalModel/src/GPU_Status.h"

#define HANDLE_CUERROR(call) {                                        \
    cudaError err = call;                                             \
    if(err != cudaSuccess) {                                          \
        errorString = std::string("CUDA error in file '") +           \
            std::string(__FILE__) + std::string("' in line ") +       \
            std::to_string(__LINE__) + std::string(": ") +            \
            std::string(cudaGetErrorString(err));                     \
        return GPU_ERROR;                                             \
    }                                                                 \
} while (0)

#define HANDLE_CUERROR_PTR(call) {                                    \
    cudaError err = call;                                             \
    if(err != cudaSuccess) {                                          \
        errorString = std::string("CUDA error in file '") +           \
            std::string(__FILE__) + std::string("' in line ") +       \
            std::to_string(__LINE__) + std::string(": ") +            \
            std::string(cudaGetErrorString(err));                     \
        return nullptr;                                               \
    }                                                                 \
} while (0)

#endif /* GPUHEADER_H */
