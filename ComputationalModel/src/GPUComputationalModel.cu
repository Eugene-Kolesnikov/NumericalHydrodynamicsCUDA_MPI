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
}

ErrorStatus GPUComputationalModel::initializeEnvironment()
{
    if(nodeType == NODE_TYPE::COMPUTATIONAL_NODE) {
        /// Initialize the scheme
        CM_HANDLE_GPUERROR(scheme->initScheme());
        /// Allocate page-locked memory for asynchronous data transferring
        /// between GPU and CPU.
        tmpCPUField = scheme->createPageLockedField(lN_X, lN_Y);
        CM_HANDLE_GPUERROR_PTR(tmpCPUField);
        lr_halo = scheme->initPageLockedHalos(2*lN_Y);
        CM_HANDLE_GPUERROR_PTR(lr_halo);
        tb_halo = scheme->initPageLockedHalos(2*lN_X);
        CM_HANDLE_GPUERROR_PTR(tb_halo);
        lrtb_halo = scheme->initPageLockedHalos(4);
        CM_HANDLE_GPUERROR_PTR(lrtb_halo);
        rcv_lr_halo = scheme->initPageLockedHalos(2*lN_Y);
        CM_HANDLE_GPUERROR_PTR(rcv_lr_halo);
        rcv_tb_halo = scheme->initPageLockedHalos(2*lN_X);
        CM_HANDLE_GPUERROR_PTR(rcv_tb_halo);
        rcv_lrtb_halo = scheme->initPageLockedHalos(4);
        CM_HANDLE_GPUERROR_PTR(rcv_lrtb_halo);
        /// Initialize CUDA streams
        HANDLE_CUERROR(cudaStreamCreate(&streamInternal));
        HANDLE_CUERROR(cudaStreamCreate(&streamHaloBorder));
        /// Allocate memory for GPU variables
        cu_field = scheme->createGPUField(lN_X, lN_Y);
        CM_HANDLE_GPUERROR_PTR(cu_field);
        cu_lr_halo = scheme->initGPUHalos(2*lN_Y);
        CM_HANDLE_GPUERROR_PTR(cu_lr_halo);
        cu_tb_halo = scheme->initGPUHalos(2*lN_X);
        CM_HANDLE_GPUERROR_PTR(cu_tb_halo);
        cu_lrtb_halo = scheme->initGPUHalos(4);
        CM_HANDLE_GPUERROR_PTR(cu_lrtb_halo);
    } else { // NODE_TYPE::SERVER_NODE
        field = scheme->createField(N_X, N_Y);
        tmpCPUField = scheme->createField(lN_X, lN_Y);
    }
    return GPU_SUCCESS;
}

ErrorStatus GPUComputationalModel::updateGlobalField(size_t mpi_node_x, size_t mpi_node_y)
{
	if(nodeType != NODE_TYPE::SERVER_NODE) {
		errorString = "GPUComputationalModel::updateGlobalField: "
			"This function should not be called by a Computational Node";
        return GPU_ERROR;
    }
	memcpyField(mpi_node_x, mpi_node_y, TmpCPUFieldToField);
    return GPU_SUCCESS;
}

ErrorStatus GPUComputationalModel::prepareSubfield(size_t mpi_node_x, size_t mpi_node_y)
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
    return GPU_SUCCESS;
}

ErrorStatus GPUComputationalModel::loadSubFieldToGPU()
{
	size_t size_of_datatype = scheme->getSizeOfDatatype();
	size_t nitems = scheme->getNumberOfElements();
	size_t sizeOfStruct = nitems * size_of_datatype;
	size_t field_size = lN_X * lN_Y;
	HANDLE_CUERROR(cudaMemcpyAsync(cu_field, tmpCPUField, field_size * sizeOfStruct, cudaMemcpyHostToDevice, streamInternal));
	*Log << std::string("Request for the stream 'streamInternal' to transfer array of ") +
		std::to_string(field_size) + std::string(" field elements from host to device has been placed.");
    return GPU_SUCCESS;
}

ErrorStatus GPUComputationalModel::gpuSync()
{
	HANDLE_CUERROR(cudaStreamSynchronize(streamInternal));
	HANDLE_CUERROR(cudaStreamSynchronize(streamHaloBorder));
	HANDLE_CUERROR(cudaDeviceSynchronize());
	*Log << "CUDA streams (GPU device) have been successfully synchronized";
    return GPU_SUCCESS;
}

ErrorStatus GPUComputationalModel::performSimulationStep()
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE) {
		errorString = "GPUComputationalModel::performSimulationStep: "
			"This function should not be called by the Server Node";
        return GPU_ERROR;
    }
	size_t CUDA_X_BLOCKS = (size_t)((double)lN_X / (double)CUDA_X_THREADS);
	size_t CUDA_Y_BLOCKS = (size_t)((double)lN_Y / (double)CUDA_Y_THREADS);
	CM_HANDLE_GPUERROR(scheme->performGPUSimulationStep(cu_field, cu_lr_halo,
        cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y,
		CUDA_X_BLOCKS, CUDA_Y_BLOCKS, CUDA_X_THREADS, CUDA_Y_THREADS,
		&streamInternal));
    return GPU_SUCCESS;
}

ErrorStatus GPUComputationalModel::updateHaloBorderElements(size_t mpi_node_x, size_t mpi_node_y)
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE) {
		errorString = "GPUComputationalModel::updateHaloBorderElements: "
			"This function should not be called by the Server Node";
        return GPU_ERROR;
    }
	size_t size_of_datatype = scheme->getSizeOfDatatype();
	size_t nitems = scheme->getNumberOfElements();
	size_t sizeOfStruct = nitems * size_of_datatype;
	/// Upload left-right halo elements
	size_t lr_size = 2 * lN_Y;
	HANDLE_CUERROR(cudaMemcpyAsync(cu_lr_halo, rcv_lr_halo, lr_size * sizeOfStruct, cudaMemcpyHostToDevice, streamHaloBorder));
	*Log << std::string("Request for the stream 'streamHaloBorder' to transfer array of ") +
        std::to_string(lr_size) + std::string(" lr_halo elements from host to device has been placed.");
	/// Upload top-bottom halo elements
	size_t tb_size = 2 * lN_X;
	HANDLE_CUERROR(cudaMemcpyAsync(cu_tb_halo, rcv_tb_halo, tb_size * sizeOfStruct, cudaMemcpyHostToDevice, streamHaloBorder));
	*Log << std::string("Request for the stream 'streamHaloBorder' to transfer array of ") +
		std::to_string(tb_size) + std::string(" tb_halo elements from host to device has been placed.");
	/// Upload diagonal halo elements
	size_t lrtb_size = 4;
	HANDLE_CUERROR(cudaMemcpyAsync(cu_lrtb_halo, rcv_lrtb_halo, lrtb_size * sizeOfStruct, cudaMemcpyHostToDevice, streamHaloBorder));
	*Log << std::string("Request for the stream 'streamHaloBorder' to transfer array of ") +
		std::to_string(lrtb_size) + std::string(" lrtb_halo (diagonal) elements from host to device has been placed.");
	/// Update global borders
	size_t CUDA_X_BLOCKS;
	if(mpi_node_x == 0) {
		/// Update global top border
		CUDA_X_BLOCKS = (size_t)(lN_X / CUDA_X_THREADS);
		CM_HANDLE_GPUERROR(scheme->updateGPUGlobalBorders(cu_field, cu_lr_halo, cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y, TOP_BORDER,
			CUDA_X_BLOCKS, 1, CUDA_X_THREADS, 1, &streamHaloBorder));
	} else if(mpi_node_x == (MPI_NODES_X - 1)) {
		/// Update global bottom border
		CUDA_X_BLOCKS = (size_t)(lN_X / CUDA_X_THREADS);
		CM_HANDLE_GPUERROR(scheme->updateGPUGlobalBorders(cu_field, cu_lr_halo, cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y, BOTTOM_BORDER,
			CUDA_X_BLOCKS, 1, CUDA_X_THREADS, 1, &streamHaloBorder));
	}
	if(mpi_node_y == 0) {
		/// Update global left border
		CUDA_X_BLOCKS = (size_t)(lN_Y / CUDA_Y_THREADS);
		CM_HANDLE_GPUERROR(scheme->updateGPUGlobalBorders(cu_field, cu_lr_halo, cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y, LEFT_BORDER,
			CUDA_X_BLOCKS, 1, CUDA_Y_THREADS, 1, &streamHaloBorder));
	} else if(mpi_node_y == (MPI_NODES_Y - 1)) {
		/// Update global right border
		CUDA_X_BLOCKS = (size_t)(lN_Y / CUDA_Y_THREADS);
		CM_HANDLE_GPUERROR(scheme->updateGPUGlobalBorders(cu_field, cu_lr_halo, cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y, RIGHT_BORDER,
			CUDA_X_BLOCKS, 1, CUDA_Y_THREADS, 1, &streamHaloBorder));
	}
	int border = -1;
	if(mpi_node_x == 0 && mpi_node_y == 0) {
		border = LEFT_TOP_BORDER;
	} else if(mpi_node_x == (MPI_NODES_X - 1) && mpi_node_y == 0) {
		border = LEFT_BOTTOM_BORDER;
	} else if(mpi_node_x == 0 && mpi_node_y == (MPI_NODES_Y - 1)) {
		border = RIGHT_TOP_BORDER;
	} else if(mpi_node_x == (MPI_NODES_X - 1) && mpi_node_y == (MPI_NODES_Y - 1)) {
		border = RIGHT_BOTTOM_BORDER;
	}
	if(border == -1) {
		/// Exit function if no diagonal element must be updated
		return GPU_SUCCESS;
	}
	/// Update a diagonal element
	CM_HANDLE_GPUERROR(scheme->updateGPUGlobalBorders(cu_field, cu_lr_halo, cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y, border,
		1, 1, 1, 1, &streamHaloBorder));
    return GPU_SUCCESS;
}

ErrorStatus GPUComputationalModel::prepareHaloElements()
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
    return GPU_SUCCESS;
}

ErrorStatus GPUComputationalModel::deinitModel()
{
    if(field != nullptr)
        delete[] (byte*)field;
    if(tmpCPUField != nullptr) {
    	if(nodeType == NODE_TYPE::COMPUTATIONAL_NODE) {
    		HANDLE_CUERROR(cudaFreeHost(tmpCPUField));
    	} else {
    		delete[] (byte*)tmpCPUField;
    	}
    }
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
    return GPU_SUCCESS;
}
