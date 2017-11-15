#include "ComputationalScheme.hpp"
#include <exception>

ComputationalScheme::ComputationalScheme()
{
}

ComputationalScheme::~ComputationalScheme()
{
}

ErrorStatus ComputationalScheme::initScheme()
{
    return initDeviceProp();
}

ErrorStatus ComputationalScheme::initDeviceProp()
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    /// Find CUDA device
    if(devCount == 0) {
        errorString = "GPUComputationalModel::initializeEnvironment: No "
            "CUDA device found!";
        return GPU_ERROR;
    }
    cudaGetDeviceProperties(&devProp, 0);
    amountSMs = devProp.multiProcessorCount;
    float architecture = devProp.major + devProp.minor / 10.0;
    totalSharedMemoryPerSM = getSheredMemoryPerSM(architecture);
    return GPU_SUCCESS;
}

size_t ComputationalScheme::getSheredMemoryPerSM(float arch)
{
    if(arch == 3.0) {
        return 49152;
    } else if(arch == 3.2) {
        return 49152;
    } else if(arch == 3.5) {
        return 49152;
    } else if(arch == 3.7) {
        return 114688;
    } else if(arch == 5.0) {
        return 65536;
    } else if(arch == 5.2) {
        return 98304;
    } else if(arch == 5.3) {
        return 65536;
    } else if(arch == 6.0) {
        return 65536;
    } else if(arch == 6.1) {
        return 98304;
    } else if(arch == 6.2) {
        return 65536;
    } else if(arch == 7.0) {
        return 98304;
    } else {
        throw std::runtime_error("ComputationalScheme: Unknown CUDA architecture!");
    }
}

std::string ComputationalScheme::getErrorString()
{
    return errorString;
}
