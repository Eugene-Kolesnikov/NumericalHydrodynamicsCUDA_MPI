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
#include <string>

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
	if(nodeType != NODE_TYPE::SERVER_NODE)
		throw std::runtime_error("GPUComputationalModel::updateGlobalField: "
				"This function should not be called by a Computational Node");
	memcpyField(mpi_node_x, mpi_node_y, TmpCPUFieldToField);
}

void GPUComputationalModel::prepareSubfield(size_t mpi_node_x, size_t mpi_node_y)
{
	if(nodeType == NODE_TYPE::COMPUTATIONAL_NODE) {
		size_t size_of_datatype = scheme->getSizeOfDatatype();
		size_t nitems = scheme->getNumberOfElements();
		size_t sizeOfStruct = nitems * size_of_datatype;
		size_t field_size = lN_X * lN_Y;
		HANDLE_CUERROR(cudaMemcpyAsync(tmpCPUField, cu_field, field_size * sizeOfStruct, cudaMemcpyDeviceToHost, streamInternal));
		*Log << std::string("Request for the stream 'streamInternal' to transfer array of ") +
					std::to_string(field_size) + std::string(" field elements from device to host has been placed.");
	} else {
		memcpyField(mpi_node_x, mpi_node_y, FieldToTmpCPUField);
	}
}

void GPUComputationalModel::loadSubFieldToGPU()
{
	size_t size_of_datatype = scheme->getSizeOfDatatype();
	size_t nitems = scheme->getNumberOfElements();
	size_t sizeOfStruct = nitems * size_of_datatype;
	size_t field_size = lN_X * lN_Y;
	HANDLE_CUERROR(cudaMemcpyAsync(cu_field, tmpCPUField, field_size * sizeOfStruct, cudaMemcpyHostToDevice, streamInternal));
	*Log << std::string("Request for the stream 'streamInternal' to transfer array of ") +
				std::to_string(field_size) + std::string(" field elements from host to device has been placed.");
}

void GPUComputationalModel::gpuSync()
{
	HANDLE_CUERROR(cudaStreamSynchronize(streamInternal));
	HANDLE_CUERROR(cudaStreamSynchronize(streamHaloBorder));
	HANDLE_CUERROR(cudaDeviceSynchronize());
	*Log << "CUDA streams (GPU device) have been successfully synchronized";
}

void GPUComputationalModel::performSimulationStep()
{
	scheme->performGPUSimulationStep(cu_field, cu_lr_halo, cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y);
}

void GPUComputationalModel::updateHaloBorderElements()
{
	size_t size_of_datatype = scheme->getSizeOfDatatype();
	size_t nitems = scheme->getNumberOfElements();
	size_t sizeOfStruct = nitems * size_of_datatype;
	size_t lr_size = 2 * lN_Y;
	HANDLE_CUERROR(cudaMemcpyAsync(cu_lr_halo, rcv_lr_halo, lr_size * sizeOfStruct, cudaMemcpyHostToDevice, streamHaloBorder));
	*Log << std::string("Request for the stream 'streamHaloBorder' to transfer array of ") +
	            std::to_string(lr_size) + std::string(" lr_halo elements from host to device has been placed.");
	size_t tb_size = 2 * lN_X;
	HANDLE_CUERROR(cudaMemcpyAsync(cu_tb_halo, rcv_tb_halo, tb_size * sizeOfStruct, cudaMemcpyHostToDevice, streamHaloBorder));
	*Log << std::string("Request for the stream 'streamHaloBorder' to transfer array of ") +
				std::to_string(tb_size) + std::string(" tb_halo elements from host to device has been placed.");
	size_t lrtb_size = 4;
	HANDLE_CUERROR(cudaMemcpyAsync(cu_lrtb_halo, rcv_lrtb_halo, lrtb_size * sizeOfStruct, cudaMemcpyHostToDevice, streamHaloBorder));
	*Log << std::string("Request for the stream 'streamHaloBorder' to transfer array of ") +
				std::to_string(lrtb_size) + std::string(" lrtb_halo (diagonal) elements from host to device has been placed.");
	/// Update global borders
	scheme->updateGPUGlobalBorders(cu_field, cu_lr_halo, cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y);
}

void GPUComputationalModel::prepareHaloElements()
{
	size_t size_of_datatype = scheme->getSizeOfDatatype();
	size_t nitems = scheme->getNumberOfElements();
	size_t sizeOfStruct = nitems * size_of_datatype;
	size_t lr_size = 2 * lN_Y;
	HANDLE_CUERROR(cudaMemcpyAsync(lr_halo, cu_lr_halo, lr_size * sizeOfStruct, cudaMemcpyDeviceToHost, streamHaloBorder));
	*Log << std::string("Request for the stream 'streamHaloBorder' to transfer array of ") +
				std::to_string(lr_size) + std::string(" lr_halo elements from device to host has been placed.");
	size_t tb_size = 2 * lN_X;
	HANDLE_CUERROR(cudaMemcpyAsync(tb_halo, cu_tb_halo, tb_size * sizeOfStruct, cudaMemcpyDeviceToHost, streamHaloBorder));
	*Log << std::string("Request for the stream 'streamHaloBorder' to transfer array of ") +
				std::to_string(tb_size) + std::string(" tb_halo elements from device to host has been placed.");
	size_t lrtb_size = 4;
	HANDLE_CUERROR(cudaMemcpyAsync(lrtb_halo, cu_lrtb_halo, lrtb_size * sizeOfStruct, cudaMemcpyDeviceToHost, streamHaloBorder));
	*Log << std::string("Request for the stream 'streamHaloBorder' to transfer array of ") +
				std::to_string(lrtb_size) + std::string(" lrtb_halo (diagonal) elements from device to host has been placed.");
}
