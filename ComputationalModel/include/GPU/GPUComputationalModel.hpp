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

#include <ComputationalModel/include/ComputationalModel.hpp>
#include <ComputationalScheme/include/GPUHeader.h>
#include <cstdlib>
#include <exception>

class GPUComputationalModel : public ComputationalModel {
public:
    GPUComputationalModel(const char* compModel, const char* gridModel);
    virtual ~GPUComputationalModel();

public:
    virtual ErrorStatus initializeEnvironment();
    virtual ErrorStatus updateGlobalField(size_t mpi_node_x, size_t mpi_node_y);
    virtual ErrorStatus prepareSubfield(size_t mpi_node_x = 0, size_t mpi_node_y = 0);
    virtual ErrorStatus loadSubFieldToGPU();
    virtual ErrorStatus gpuSync();
    virtual ErrorStatus performSimulationStep();
    virtual ErrorStatus updateHaloBorderElements(size_t mpi_node_x, size_t mpi_node_y);
    virtual ErrorStatus prepareHaloElements();
    virtual ErrorStatus deinitModel();

protected:
    ErrorStatus updateGPUHaloElements(size_t lr_size, size_t tb_size, size_t lrtb_size);
#ifdef __DEBUG__
    ErrorStatus updateDBGHaloElements(size_t lr_size, size_t tb_size, size_t lrtb_size);
#endif

protected:
    cudaStream_t streamInternal;
    cudaStream_t streamHaloBorder;

protected:
    void* cu_field;
    void* cu_lr_halo;
    void* cu_tb_halo;
    void* cu_lrtb_halo;
    void* snd_cu_lr_halo;
    void* snd_cu_tb_halo;
    void* snd_cu_lrtb_halo;

protected:
    cudaDeviceProp devProp;
    cudaError_t lastCudaError;
    size_t amountSMs;
    size_t totalSharedMemoryPerSM;
};

#endif /* GPUCOMPUTATIONALMODEL_HPP */
