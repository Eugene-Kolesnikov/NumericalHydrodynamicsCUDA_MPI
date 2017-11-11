/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   LatticeBoltzmannModel.hpp
 * Author: eugene
 *
 * Created on November 1, 2017, 2:20 PM
 */

#ifndef GPUCOMPUTATIONALMODEL_HPP
#define GPUCOMPUTATIONALMODEL_HPP

#include "ComputationalModel.hpp"
#include <cstdlib>
#include <cuda.h>
#include <exception>

#define HANDLE_CUERROR(call) {										             \
    cudaError err = call;												         \
    if(err != cudaSuccess) {
        std::string error = std::string("CUDA error in file '") + __FILE__ +  \
                "' in line " + __LINE__ + ": " + cudaGetErrorString(err);     \
        throw std::runtime_error(error);                                      \                          \
    }                                                                         \
} while (0)

class GPUComputationalModel : public ComputationalModel {
public:
    GPUComputationalModel();
    virtual ~GPUComputationalModel();
    
public:
    virtual void initializeField();
    virtual void updateGlobalField(size_t mpi_node_x, size_t mpi_node_y);
    virtual void prepareSubfield(size_t mpi_node_x = 0, size_t mpi_node_y = 0);
    virtual void loadSubFieldToGPU();
    virtual void gpuSync();
    virtual void performSimulationStep();
    virtual void updateHaloBorderElements();
    virtual void prepareHaloElements();
    
protected:
    cudaStream_t streamInternal;
    cudaStream_t streamHaloBorder;
    
protected:
    void* cu_field;
    void* cu_lr_halo;
    void* cu_tb_halo;
    void* cu_lrtb_halo;
};

#endif /* GPUCOMPUTATIONALMODEL_HPP */

