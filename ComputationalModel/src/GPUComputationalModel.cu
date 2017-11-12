/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   LatticeBoltzmannModel.cpp
 * Author: eugene
 * 
 * Created on November 1, 2017, 2:20 PM
 */

#include "GPUComputationalModel.hpp"

GPUComputationalModel::GPUComputationalModel(const char* compModel, const char* gridModel):
    ComputationalModel(compModel, gridModel)
{
    cu_field = nullptr;
    cu_lr_halo = nullptr;
    cu_tb_halo = nullptr;
    cu_lrtb_halo = nullptr;
}

GPUComputationalModel::~GPUComputationalModel() 
{
    if(field != nullptr)
        HANDLE_CUERROR(cudaFreeHost((byte*)field));
    if(tmpCPUField != nullptr)
        HANDLE_CUERROR(cudaFreeHost((byte*)tmpCPUField));
    if(lr_halo != nullptr)
        HANDLE_CUERROR(cudaFreeHost((byte*)lr_halo));
    if(tb_halo != nullptr)
        HANDLE_CUERROR(cudaFreeHost((byte*)tb_halo));
    if(lrtb_halo != nullptr)
        HANDLE_CUERROR(cudaFreeHost((byte*)lrtb_halo));
    if(rcv_lr_halo != nullptr)
        HANDLE_CUERROR(cudaFreeHost((byte*)rcv_lr_halo));
    if(rcv_tb_halo != nullptr)
        HANDLE_CUERROR(cudaFreeHost((byte*)rcv_tb_halo));
    if(rcv_lrtb_halo != nullptr)
        HANDLE_CUERROR(cudaFreeHost((byte*)rcv_lrtb_halo));
    if(cu_field != nullptr)
        HANDLE_CUERROR(cudaFree((byte*)cu_field));
    if(cu_lr_halo != nullptr)
        HANDLE_CUERROR(cudaFree((byte*)cu_lr_halo));
    if(cu_tb_halo != nullptr)
        HANDLE_CUERROR(cudaFree((byte*)cu_tb_halo));
    if(cu_lrtb_halo != nullptr)
        HANDLE_CUERROR(cudaFree((byte*)cu_lrtb_halo));
}

void GPUComputationalModel::initializeEnvironment()
{
    if(nodeType == NODE_TYPE::COMPUTATIONAL_NODE) {
        /// Allocate page-locked memory for asynchronous data transferring
        /// between GPU and CPU.
        tmpCPUField = scheme->createPageLockedField(lN_X, lN_Y);
        lr_halo = scheme->initPageLockedHalos(2*lN_Y);
        tb_halo = scheme->initPageLockedHalos(2*lN_X);
        lrtb_halo = scheme->initPageLockedHalos(4);
        rcv_lr_halo = scheme->initPageLockedHalos(2*lN_Y);
        rcv_tb_halo = scheme->initPageLockedHalos(2*lN_X);
        rcv_lrtb_halo = scheme->initPageLockedHalos(4);
        /// Initialize CUDA streams
        cudaStreamCreate(&streamInternal);
        cudaStreamCreate(&streamHaloBorder);
        /// Allocate memory for GPU variables
        cu_field = scheme->createGPUField(lN_X, lN_Y);
        cu_lr_halo = scheme->initGPUHalos(2*lN_Y);
        cu_tb_halo = scheme->initGPUHalos(2*lN_X);
        cu_lrtb_halo = scheme->initGPUHalos(4);
    } else { // NODE_TYPE::SERVER_NODE
        field = scheme->createField(N_X, N_Y);
        tmpCPUField = scheme->createField(lN_X, lN_Y);
    }
}

void GPUComputationalModel::updateGlobalField(size_t mpi_node_x, size_t mpi_node_y)
{
    
}

void GPUComputationalModel::prepareSubfield(size_t mpi_node_x, size_t mpi_node_y)
{
    
}

void GPUComputationalModel::loadSubFieldToGPU()
{
    
}

void GPUComputationalModel::gpuSync()
{
    
}

void GPUComputationalModel::performSimulationStep()
{
    
}

void GPUComputationalModel::updateHaloBorderElements()
{
    
}

void GPUComputationalModel::prepareHaloElements()
{
    
}
