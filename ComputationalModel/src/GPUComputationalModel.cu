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

typedef char byte;

GPUComputationalModel::GPUComputationalModel(const char* compModel, const char* gridModel):
    ComputationalModel(compModel, gridModel)
{
    cu_field = nullptr;
    cu_lr_halo = nullptr;
    cu_tb_halo = nullptr;
    cu_lrtb_halo = nullptr;
    snd_cu_lr_halo = nullptr;
    snd_cu_tb_halo = nullptr;
    snd_cu_lrtb_halo = nullptr;
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
        #ifdef __DEBUG__
            dbg_field = scheme->createPageLockedField(lN_X, lN_Y);
        #endif
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

        snd_cu_lr_halo = scheme->initGPUHalos(2*lN_Y);
        CM_HANDLE_GPUERROR_PTR(snd_cu_lr_halo);
        snd_cu_tb_halo = scheme->initGPUHalos(2*lN_X);
        CM_HANDLE_GPUERROR_PTR(snd_cu_tb_halo);
        snd_cu_lrtb_halo = scheme->initGPUHalos(4);
        CM_HANDLE_GPUERROR_PTR(snd_cu_lrtb_halo);
        #ifdef __DEBUG__
            dbg_lr_halo = scheme->initPageLockedHalos(2*lN_Y);
            dbg_tb_halo = scheme->initPageLockedHalos(2*lN_X);
            dbg_lrtb_halo = scheme->initPageLockedHalos(4);
        #endif
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
		size_t sizeOfStruct = scheme->getSizeOfDatastruct();
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
	size_t sizeOfStruct = scheme->getSizeOfDatastruct();
	size_t field_size = lN_X * lN_Y;
	HANDLE_CUERROR(cudaMemcpyAsync(cu_field, tmpCPUField, field_size * sizeOfStruct, cudaMemcpyHostToDevice, streamInternal));
	*Log << std::string("Request for the stream 'streamInternal' to transfer array of ") +
		std::to_string(field_size) + std::string(" field elements from host to device has been placed.");
    #ifdef __DEBUG__
        cpu_memcpy(dbg_field, tmpCPUField, field_size * sizeOfStruct);
        CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_field, cu_field, 0, field_size * sizeOfStruct, *Log);
    #endif
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
    #ifdef __DEBUG__
        {
            size_t numberOfBytes = 2*lN_Y * scheme->getSizeOfDatastruct();
            CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_lr_halo, cu_lr_halo, 0, numberOfBytes, *Log);
        }
    #endif
	CM_HANDLE_GPUERROR(scheme->performGPUSimulationStep(cu_field, cu_lr_halo,
        cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y, CUDA_X_THREADS, CUDA_Y_THREADS,
		&streamInternal));
    #ifdef __DEBUG__
        size_t numberOfBytes = lN_X * lN_Y * scheme->getSizeOfDatastruct();
        scheme->dbg_performSimulationStep(dbg_field, dbg_lr_halo,
            dbg_tb_halo, dbg_lrtb_halo, lN_X, lN_Y, CUDA_X_THREADS, CUDA_Y_THREADS,
    		&streamInternal);
        /**********************************************************************/
        //HANDLE_CUERROR(cudaMemcpyAsync(cu_field, dbg_field, lN_X*lN_Y * scheme->getSizeOfDatastruct(), cudaMemcpyHostToDevice, streamInternal));
        //gpuSync();
        /**********************************************************************/
        CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_field, cu_field, 0, numberOfBytes, *Log);
        //_DEBUG_PRINT_FIELDS_CPU_GPU(dbg_field, cu_field, lN_X, lN_Y, scheme->getSizeOfDatastruct(), *Log);
    #endif
    return GPU_SUCCESS;
}

ErrorStatus GPUComputationalModel::updateHaloBorderElements(size_t mpi_node_x, size_t mpi_node_y)
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE) {
		errorString = "GPUComputationalModel::updateHaloBorderElements: "
			"This function should not be called by the Server Node";
        return GPU_ERROR;
    }
    #ifdef __DEBUG__
        //_DEBUG_PRINT_FIELDS_CPU_GPU(dbg_field, cu_field, lN_X, lN_Y, scheme->getSizeOfDatastruct(), *Log);
    #endif
	size_t sizeOfStruct = scheme->getSizeOfDatastruct();
	/// Upload left-right halo elements
	size_t lr_size = 2 * lN_Y;
	HANDLE_CUERROR(cudaMemcpyAsync(cu_lr_halo, rcv_lr_halo, lr_size * sizeOfStruct, cudaMemcpyHostToDevice, streamHaloBorder));
	*Log << std::string("Request for the stream 'streamHaloBorder' to transfer array of ") +
        std::to_string(lr_size) + std::string(" lr_halo elements from host to device has been placed.");
    #ifdef __DEBUG__
        cpu_memcpy(dbg_lr_halo, rcv_lr_halo, lr_size);
        CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_lr_halo, cu_lr_halo, 0, lr_size * sizeOfStruct, *Log);
    #endif
	/// Upload top-bottom halo elements
	size_t tb_size = 2 * lN_X;
	HANDLE_CUERROR(cudaMemcpyAsync(cu_tb_halo, rcv_tb_halo, tb_size * sizeOfStruct, cudaMemcpyHostToDevice, streamHaloBorder));
	*Log << std::string("Request for the stream 'streamHaloBorder' to transfer array of ") +
		std::to_string(tb_size) + std::string(" tb_halo elements from host to device has been placed.");
    #ifdef __DEBUG__
        cpu_memcpy(dbg_tb_halo, rcv_tb_halo, tb_size);
        CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_tb_halo, cu_tb_halo, 0, tb_size * sizeOfStruct, *Log);
    #endif
	/// Upload diagonal halo elements
	size_t lrtb_size = 4;
	HANDLE_CUERROR(cudaMemcpyAsync(cu_lrtb_halo, rcv_lrtb_halo, lrtb_size * sizeOfStruct, cudaMemcpyHostToDevice, streamHaloBorder));
	*Log << std::string("Request for the stream 'streamHaloBorder' to transfer array of ") +
		std::to_string(lrtb_size) + std::string(" lrtb_halo (diagonal) elements from host to device has been placed.");
    #ifdef __DEBUG__
        cpu_memcpy(dbg_lrtb_halo, rcv_lrtb_halo, lrtb_size);
        CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_lrtb_halo, cu_lrtb_halo, 0, lrtb_size * sizeOfStruct, *Log);
    #endif
	/// Update global borders
	if(mpi_node_x == 0) {
		/// Update global left border
		CM_HANDLE_GPUERROR(scheme->updateGPUGlobalBorders(cu_field, cu_lr_halo,
            cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y, CU_LEFT_BORDER,
			CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder));
        #ifdef __DEBUG__
            size_t numberOfBytes = lN_Y * sizeOfStruct;
            scheme->dbg_updateGlobalBorders(dbg_field, dbg_lr_halo,
                dbg_tb_halo, dbg_lrtb_halo, lN_X, lN_Y, CU_LEFT_BORDER,
    			CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder);
            CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_lr_halo, cu_lr_halo, 0, numberOfBytes, *Log);
        #endif
	}
    if(mpi_node_x == (MPI_NODES_X - 1)) {
		/// Update global right border
		CM_HANDLE_GPUERROR(scheme->updateGPUGlobalBorders(cu_field, cu_lr_halo,
            cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y, CU_RIGHT_BORDER,
			CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder));
        #ifdef __DEBUG__
            size_t numberOfBytes = lN_Y * sizeOfStruct;
            scheme->dbg_updateGlobalBorders(dbg_field, dbg_lr_halo,
                dbg_tb_halo, dbg_lrtb_halo, lN_X, lN_Y, CU_RIGHT_BORDER,
    			CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder);
            CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_lr_halo, cu_lr_halo,
                lN_Y * sizeOfStruct, numberOfBytes, *Log);
        #endif
	}
	if(mpi_node_y == 0) {
		/// Update global top border
		CM_HANDLE_GPUERROR(scheme->updateGPUGlobalBorders(cu_field, cu_lr_halo,
            cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y, CU_TOP_BORDER,
			CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder));
        #ifdef __DEBUG__
            size_t numberOfBytes = lN_X * sizeOfStruct;
            scheme->dbg_updateGlobalBorders(dbg_field, dbg_lr_halo,
                dbg_tb_halo, dbg_lrtb_halo, lN_X, lN_Y, CU_TOP_BORDER,
    			CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder);
            CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_tb_halo, cu_tb_halo, 0,
                numberOfBytes, *Log);
        #endif
	}
    if(mpi_node_y == (MPI_NODES_Y - 1)) {
		/// Update global bottom border
		CM_HANDLE_GPUERROR(scheme->updateGPUGlobalBorders(cu_field, cu_lr_halo,
            cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y, CU_BOTTOM_BORDER,
			CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder));
        #ifdef __DEBUG__
            size_t numberOfBytes = lN_X * sizeOfStruct;
            scheme->dbg_updateGlobalBorders(dbg_field, dbg_lr_halo,
                dbg_tb_halo, dbg_lrtb_halo, lN_X, lN_Y, CU_BOTTOM_BORDER,
    			CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder);
            CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_tb_halo, cu_tb_halo,
                lN_X * sizeOfStruct, numberOfBytes, *Log);
        #endif
	}
    /// Update global diagonal border elements
    if(mpi_node_y == 0) {
        /// First row of nodes have the top global border which means that
        /// they have to update top diagonal border elements
        /// Update the left-top diagonal element
    	CM_HANDLE_GPUERROR(scheme->updateGPUGlobalBorders(cu_field, cu_lr_halo,
            cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y, CU_LEFT_TOP_BORDER,
    		CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder));
        #ifdef __DEBUG__
            scheme->dbg_updateGlobalBorders(dbg_field, dbg_lr_halo,
                dbg_tb_halo, dbg_lrtb_halo, lN_X, lN_Y, CU_LEFT_TOP_BORDER,
        		CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder);
            CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_lrtb_halo, cu_lrtb_halo,
                (CU_LEFT_TOP_BORDER - CU_LEFT_TOP_BORDER) * sizeOfStruct, sizeOfStruct, *Log);
        #endif
        /// Update the right-top diagonal element
        CM_HANDLE_GPUERROR(scheme->updateGPUGlobalBorders(cu_field, cu_lr_halo,
            cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y, CU_RIGHT_TOP_BORDER,
    		CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder));
        #ifdef __DEBUG__
            scheme->dbg_updateGlobalBorders(dbg_field, dbg_lr_halo,
                dbg_tb_halo, dbg_lrtb_halo, lN_X, lN_Y, CU_RIGHT_TOP_BORDER,
        		CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder);
            CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_lrtb_halo, cu_lrtb_halo,
                (CU_RIGHT_TOP_BORDER - CU_LEFT_TOP_BORDER) * sizeOfStruct, sizeOfStruct, *Log);
        #endif
    }
    if(mpi_node_y == (MPI_NODES_Y - 1)) {
        /// Last row of nodes have the bottom global border which means that
        /// they have to update bottom diagonal border elements
        /// Update the left-bottom diagonal element
    	CM_HANDLE_GPUERROR(scheme->updateGPUGlobalBorders(cu_field, cu_lr_halo,
            cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y, CU_LEFT_BOTTOM_BORDER,
    		CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder));
        #ifdef __DEBUG__
            scheme->dbg_updateGlobalBorders(dbg_field, dbg_lr_halo,
                dbg_tb_halo, dbg_lrtb_halo, lN_X, lN_Y, CU_LEFT_BOTTOM_BORDER,
        		CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder);
            CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_lrtb_halo, cu_lrtb_halo,
                (CU_LEFT_BOTTOM_BORDER - CU_LEFT_TOP_BORDER) * sizeOfStruct, sizeOfStruct, *Log);
        #endif
        /// Update the right-bottom diagonal element
        CM_HANDLE_GPUERROR(scheme->updateGPUGlobalBorders(cu_field, cu_lr_halo,
            cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y, CU_RIGHT_BOTTOM_BORDER,
    		CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder));
        #ifdef __DEBUG__
            scheme->dbg_updateGlobalBorders(dbg_field, dbg_lr_halo,
                dbg_tb_halo, dbg_lrtb_halo, lN_X, lN_Y, CU_RIGHT_BOTTOM_BORDER,
        		CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder);
            CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_lrtb_halo, cu_lrtb_halo,
                (CU_RIGHT_BOTTOM_BORDER - CU_LEFT_TOP_BORDER) * sizeOfStruct, sizeOfStruct, *Log);
        #endif
    }
    if(mpi_node_x == 0) {
        /// First column of nodes have the left global border which means that
        /// they have to update left-top and left-bottom diagonal border elements
        /// Update the left-top diagonal element
        if(mpi_node_y != 0) {
            /// Since the first block has already updated the left-top diagonal element
            CM_HANDLE_GPUERROR(scheme->updateGPUGlobalBorders(cu_field, cu_lr_halo,
                cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y, CU_LEFT_TOP_BORDER,
        		CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder));
            #ifdef __DEBUG__
                scheme->dbg_updateGlobalBorders(dbg_field, dbg_lr_halo,
                    dbg_tb_halo, dbg_lrtb_halo, lN_X, lN_Y, CU_LEFT_TOP_BORDER,
            		CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder);
                CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_lrtb_halo, cu_lrtb_halo,
                    (CU_LEFT_TOP_BORDER - CU_LEFT_TOP_BORDER) * sizeOfStruct, sizeOfStruct, *Log);
            #endif
        }
        if(mpi_node_y != (MPI_NODES_Y - 1)) {
            /// Since the last block has already updated the left-bottom diagonal element
            CM_HANDLE_GPUERROR(scheme->updateGPUGlobalBorders(cu_field, cu_lr_halo,
                cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y, CU_LEFT_BOTTOM_BORDER,
        		CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder));
            #ifdef __DEBUG__
                scheme->dbg_updateGlobalBorders(dbg_field, dbg_lr_halo,
                    dbg_tb_halo, dbg_lrtb_halo, lN_X, lN_Y, CU_LEFT_BOTTOM_BORDER,
            		CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder);
                CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_lrtb_halo, cu_lrtb_halo,
                    (CU_LEFT_BOTTOM_BORDER - CU_LEFT_TOP_BORDER) * sizeOfStruct, sizeOfStruct, *Log);
            #endif
        }
    }
    if(mpi_node_x == (MPI_NODES_X - 1)) {
        /// Last column of nodes have the right global border which means that
        /// they have to update right-top and right-bottom diagonal border elements
        /// Update the right-top diagonal element
        if(mpi_node_y != 0) {
            /// Since the first block has already updated the right-top diagonal element
            CM_HANDLE_GPUERROR(scheme->updateGPUGlobalBorders(cu_field, cu_lr_halo,
                cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y, CU_RIGHT_TOP_BORDER,
        		CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder));
            #ifdef __DEBUG__
                scheme->dbg_updateGlobalBorders(dbg_field, dbg_lr_halo,
                    dbg_tb_halo, dbg_lrtb_halo, lN_X, lN_Y, CU_RIGHT_TOP_BORDER,
            		CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder);
                CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_lrtb_halo, cu_lrtb_halo,
                    (CU_RIGHT_TOP_BORDER - CU_LEFT_TOP_BORDER) * sizeOfStruct, sizeOfStruct, *Log);
            #endif
        }
        /// Update the right-bottom diagonal element
        if(mpi_node_y != (MPI_NODES_Y - 1)) {
            /// Since the last block has already updated the right-bottom diagonal element
            CM_HANDLE_GPUERROR(scheme->updateGPUGlobalBorders(cu_field, cu_lr_halo,
                cu_tb_halo, cu_lrtb_halo, lN_X, lN_Y, CU_RIGHT_BOTTOM_BORDER,
        		CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder));
            #ifdef __DEBUG__
                scheme->dbg_updateGlobalBorders(dbg_field, dbg_lr_halo,
                    dbg_tb_halo, dbg_lrtb_halo, lN_X, lN_Y, CU_RIGHT_BOTTOM_BORDER,
            		CUDA_X_THREADS, CUDA_Y_THREADS, &streamHaloBorder);
                CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_lrtb_halo, cu_lrtb_halo,
                    (CU_RIGHT_BOTTOM_BORDER - CU_LEFT_TOP_BORDER) * sizeOfStruct, sizeOfStruct, *Log);
            #endif
        }
    }
    #ifdef __DEBUG__
    {
        size_t sizeOfStruct = scheme->getSizeOfDatastruct();
        CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_lr_halo, cu_lr_halo, 0, 2*lN_Y*sizeOfStruct, *Log);
        //_DEBUG_PRINT_ARRAYS_CPU_GPU(dbg_lr_halo, cu_lr_halo, 2*lN_Y, sizeOfStruct, *Log);
        CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_tb_halo, cu_tb_halo, 0, 2*lN_X*sizeOfStruct, *Log);
        //_DEBUG_PRINT_ARRAYS_CPU_GPU(dbg_tb_halo, cu_tb_halo, 2*lN_X, sizeOfStruct, *Log);
        CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_lrtb_halo, cu_lrtb_halo, 0, 4*sizeOfStruct, *Log);
        //_DEBUG_PRINT_ARRAYS_CPU_GPU(dbg_lrtb_halo, cu_lrtb_halo, 4, sizeOfStruct, *Log);
    }
    #endif
    return GPU_SUCCESS;
}

ErrorStatus GPUComputationalModel::prepareHaloElements()
{
    #ifdef __DEBUG__
        //_DEBUG_PRINT_FIELDS_CPU_GPU(dbg_field, cu_field, lN_X, lN_Y, scheme->getSizeOfDatastruct(), *Log);
    #endif
	size_t sizeOfStruct = scheme->getSizeOfDatastruct();
	size_t lr_size = 2 * lN_Y;
    size_t tb_size = 2 * lN_X;
    size_t lrtb_size = 4;
    /// Update the halo elements on the GPU
    CM_HANDLE_GPUERROR(updateGPUHaloElements(lr_size, tb_size, lrtb_size));
    /// Send requests to transfer updated halos from the GPU to the CPU
	HANDLE_CUERROR(cudaMemcpyAsync(lr_halo, snd_cu_lr_halo, lr_size * sizeOfStruct, cudaMemcpyDeviceToHost, streamHaloBorder));
	*Log << std::string("Request for the stream 'streamHaloBorder' to transfer array of ") +
		std::to_string(lr_size) + std::string(" lr_halo elements from device to host has been placed.");
    #ifdef __DEBUG__
        HANDLE_CUERROR(cudaDeviceSynchronize());
        CHECK_CPU_CPU_ARRAYS_EQUALITY_BYTES(lr_halo, dbg_lr_halo, lr_size * sizeOfStruct, *Log);
        //_DEBUG_PRINT_ARRAYS_CPU_CPU(dbg_lr_halo, lr_halo, lr_size, sizeOfStruct, *Log);
    #endif
	HANDLE_CUERROR(cudaMemcpyAsync(tb_halo, snd_cu_tb_halo, tb_size * sizeOfStruct, cudaMemcpyDeviceToHost, streamHaloBorder));
	*Log << std::string("Request for the stream 'streamHaloBorder' to transfer array of ") +
		std::to_string(tb_size) + std::string(" tb_halo elements from device to host has been placed.");
    #ifdef __DEBUG__
        HANDLE_CUERROR(cudaDeviceSynchronize());
        CHECK_CPU_CPU_ARRAYS_EQUALITY_BYTES(tb_halo, dbg_tb_halo, tb_size * sizeOfStruct, *Log);
        //_DEBUG_PRINT_ARRAYS_CPU_CPU(dbg_tb_halo, tb_halo, tb_size, sizeOfStruct, *Log);
    #endif
	HANDLE_CUERROR(cudaMemcpyAsync(lrtb_halo, snd_cu_lrtb_halo, lrtb_size * sizeOfStruct, cudaMemcpyDeviceToHost, streamHaloBorder));
	*Log << std::string("Request for the stream 'streamHaloBorder' to transfer array of ") +
		std::to_string(lrtb_size) + std::string(" lrtb_halo (diagonal) elements from device to host has been placed.");
    #ifdef __DEBUG__
        HANDLE_CUERROR(cudaDeviceSynchronize());
        CHECK_CPU_CPU_ARRAYS_EQUALITY_BYTES(lrtb_halo, dbg_lrtb_halo, lrtb_size * sizeOfStruct, *Log);
        //_DEBUG_PRINT_ARRAYS_CPU_CPU(dbg_lrtb_halo, lrtb_halo, lrtb_size, sizeOfStruct, *Log);
    #endif
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
    if(snd_cu_lr_halo != nullptr)
        HANDLE_CUERROR(cudaFree((byte*)snd_cu_lr_halo));
    if(snd_cu_tb_halo != nullptr)
        HANDLE_CUERROR(cudaFree((byte*)snd_cu_tb_halo));
    if(snd_cu_lrtb_halo != nullptr)
        HANDLE_CUERROR(cudaFree((byte*)snd_cu_lrtb_halo));
    #ifdef __DEBUG__
        HANDLE_CUERROR(cudaFreeHost((byte*)dbg_field));
        HANDLE_CUERROR(cudaFreeHost((byte*)dbg_lr_halo));
        HANDLE_CUERROR(cudaFreeHost((byte*)dbg_tb_halo));
        HANDLE_CUERROR(cudaFreeHost((byte*)dbg_lrtb_halo));
    #endif
    return GPU_SUCCESS;
}

__device__ void copyHaloData(byte* to, byte* from, size_t sizeOfStruct)
{
    for(size_t i = 0; i < sizeOfStruct; ++i) {
        to[i] = from[i];
    }
}

/*__global__ void updateGPUHaloElements_kernel(byte* cu_field, byte* snd_cu_lr_halo,
    byte* snd_cu_tb_halo, byte* snd_cu_lrtb_halo, size_t N_X, size_t N_Y, size_t lr_size,
    size_t tb_size, size_t lrtb_size, size_t lr_id, size_t tb_id,
    size_t lrtb_id, size_t totalThreads, size_t sizeOfStruct)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= totalThreads)
        return;
    byte* to = nullptr;
    byte* from = nullptr;
    size_t offset;
    if(tid > lrtb_id) {
        /// Diagonal elements
        // '+ CU_LEFT_TOP_BORDER' is needed because it has a value 4
        tid = tid - lrtb_id + CU_LEFT_TOP_BORDER;
        if(tid == CU_LEFT_TOP_BORDER) {
            to = snd_cu_lrtb_halo + CU_LEFT_TOP_BORDER * sizeOfStruct;
            from = cu_field;
        } else if(tid == CU_RIGHT_TOP_BORDER) {
            to = snd_cu_lrtb_halo + CU_RIGHT_TOP_BORDER * sizeOfStruct;
            from = cu_field + (N_X - 1) * sizeOfStruct;
        } else if(tid == CU_LEFT_BOTTOM_BORDER) {
            to = snd_cu_lrtb_halo + CU_LEFT_BOTTOM_BORDER * sizeOfStruct;
            from = cu_field + N_X * (N_Y - 1) * sizeOfStruct;
        } else { // tid == CU_RIGHT_BOTTOM_BORDER
            to = snd_cu_lrtb_halo + CU_RIGHT_BOTTOM_BORDER * sizeOfStruct;
            from = cu_field + (N_X * N_Y - 1) * sizeOfStruct;
        }
    } else if(tid > tb_id) {
        /// Top-Bottom elements
        tid -= tb_id;
        offset = tid < N_X ? 0 : (N_Y-1) * N_X * sizeOfStruct;
        to = snd_cu_tb_halo + tid * sizeOfStruct;
        from = cu_field + offset + tid * sizeOfStruct;
    } else {
        /// Left-Right elements
        offset = tid < N_Y ? tid * N_X * sizeOfStruct : ((tid + 1) * N_X - 1) * sizeOfStruct;
        to = snd_cu_lr_halo + tid * sizeOfStruct;
        from = cu_field + offset;
    }
    copyHaloData(to, from, sizeOfStruct);
}*/

// TODO: Make a faster version of 'updateGPUHaloElements_kernel'
// Probably, separate computation of each border. Most likely, improve the above version.
__global__ void updateGPUHaloElements_kernel(byte* cu_field, byte* snd_cu_lr_halo,
    byte* snd_cu_tb_halo, byte* snd_cu_lrtb_halo, size_t N_X, size_t N_Y, size_t lr_size,
    size_t tb_size, size_t lrtb_size, size_t totalThreads, size_t sizeOfStruct)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= totalThreads)
        return;
    byte* to = nullptr;
    byte* from = nullptr;
    if(tid < lr_size) {
        to = snd_cu_lr_halo + tid * sizeOfStruct;
        from = cu_field + tid * N_X * sizeOfStruct;
        copyHaloData(to, from, sizeOfStruct);
        to = snd_cu_lr_halo + (N_Y + tid) * sizeOfStruct;
        from = cu_field + ((tid + 1) * N_X - 1) * sizeOfStruct;
        copyHaloData(to, from, sizeOfStruct);
    }
    if(tid < tb_size) {
        to = snd_cu_tb_halo + tid * sizeOfStruct;
        from = cu_field + tid * sizeOfStruct;
        copyHaloData(to, from, sizeOfStruct);
        to = snd_cu_tb_halo + (N_X + tid) * sizeOfStruct;
        from = cu_field + ((N_Y - 1) * N_X + tid) * sizeOfStruct;
        copyHaloData(to, from, sizeOfStruct);
    }
    if(tid < lrtb_size) {
        // '+ CU_LEFT_TOP_BORDER' is needed because it has a value 4
        tid = tid + CU_LEFT_TOP_BORDER;
        if(tid == CU_LEFT_TOP_BORDER) {
            to = snd_cu_lrtb_halo + (tid - CU_LEFT_TOP_BORDER) * sizeOfStruct;
            from = cu_field;
        } else if(tid == CU_RIGHT_TOP_BORDER) {
            to = snd_cu_lrtb_halo + (tid - CU_LEFT_TOP_BORDER) * sizeOfStruct;
            from = cu_field + (N_X - 1) * sizeOfStruct;
        } else if(tid == CU_LEFT_BOTTOM_BORDER) {
            to = snd_cu_lrtb_halo + (tid - CU_LEFT_TOP_BORDER) * sizeOfStruct;
            from = cu_field + N_X * (N_Y - 1) * sizeOfStruct;
        } else { // tid == CU_RIGHT_BOTTOM_BORDER
            to = snd_cu_lrtb_halo + (tid - CU_LEFT_TOP_BORDER) * sizeOfStruct;
            from = cu_field + (N_X * N_Y - 1) * sizeOfStruct;
        }
        copyHaloData(to, from, sizeOfStruct);
    }
}

ErrorStatus GPUComputationalModel::updateGPUHaloElements(size_t lr_size,
    size_t tb_size, size_t lrtb_size)
{
    #ifdef __DEBUG__
        //_DEBUG_PRINT_FIELDS_CPU_GPU(dbg_field, cu_field, lN_X, lN_Y, scheme->getSizeOfDatastruct(), *Log);
    #endif
    size_t TotalThreads = lr_size > tb_size ? (lr_size > lrtb_size ? lr_size : lrtb_size)
                                            : (tb_size > lrtb_size ? tb_size : lrtb_size);
    size_t denom = lr_size > tb_size ? (lr_size > lrtb_size ? CUDA_Y_THREADS : 4)
                                     : (tb_size > lrtb_size ? CUDA_X_THREADS : 4);
    size_t CUDA_X_BLOCKS = (size_t)(floor(((double)TotalThreads-1) / denom + 1));
    //streamHaloBorder
    /// Launch the CUDA kernel
	/*updateGPUHaloElements_kernel <<< dim3(CUDA_X_BLOCKS, 1, 1),
		dim3(CUDA_X_THREADS, 1, 1), 0, streamHaloBorder >>>
            ((byte*)cu_field, (byte*)snd_cu_lr_halo, (byte*)snd_cu_tb_halo,
            (byte*)snd_cu_lrtb_halo, lN_X, lN_Y, lr_size, tb_size, lrtb_size,
            0, lr_size, lr_size + tb_size, TotalThreads, scheme->getSizeOfDatastruct());*/
    updateGPUHaloElements_kernel <<< dim3(CUDA_X_BLOCKS, 1, 1),
		dim3(denom, 1, 1), 0, streamHaloBorder >>>
            ((byte*)cu_field, (byte*)snd_cu_lr_halo, (byte*)snd_cu_tb_halo,
            (byte*)snd_cu_lrtb_halo, lN_X, lN_Y, lr_size, tb_size, lrtb_size,
            TotalThreads, scheme->getSizeOfDatastruct());
    /// Check if the kernel executed without errors
    lastCudaError = cudaGetLastError();
    if(lastCudaError != cudaSuccess) {
        errorString = std::string("updateGPUGlobalBorders: ") +
            std::string(cudaGetErrorString(lastCudaError));
        return GPU_ERROR;
    }
    #ifdef __DEBUG__
        size_t sizeOfStruct = scheme->getSizeOfDatastruct();
        CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_field, cu_field, 0, lN_X * lN_Y * sizeOfStruct, *Log);
        updateDBGHaloElements(lr_size, tb_size, lrtb_size);
        /**********************************************************************/
        //HANDLE_CUERROR(cudaMemcpyAsync(snd_cu_lr_halo, dbg_lr_halo, lr_size * sizeOfStruct, cudaMemcpyHostToDevice, streamHaloBorder));
        //HANDLE_CUERROR(cudaMemcpyAsync(snd_cu_tb_halo, dbg_tb_halo, tb_size * sizeOfStruct, cudaMemcpyHostToDevice, streamHaloBorder));
        //HANDLE_CUERROR(cudaMemcpyAsync(snd_cu_lrtb_halo, dbg_lrtb_halo, lrtb_size * sizeOfStruct, cudaMemcpyHostToDevice, streamHaloBorder));
        //gpuSync();
        /**********************************************************************/
        //_DEBUG_PRINT_ARRAYS_CPU_GPU(dbg_lr_halo, snd_cu_lr_halo, lr_size, sizeOfStruct, *Log);
        CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_lr_halo, snd_cu_lr_halo, 0, lr_size * sizeOfStruct, *Log);
        CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_tb_halo, snd_cu_tb_halo, 0, tb_size * sizeOfStruct, *Log);
        CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(dbg_lrtb_halo, snd_cu_lrtb_halo, 0, lrtb_size * sizeOfStruct, *Log);
    #endif
    return GPU_SUCCESS;
}

#ifdef __DEBUG__
ErrorStatus GPUComputationalModel::updateDBGHaloElements(size_t lr_size,
    size_t tb_size, size_t lrtb_size)
{
    size_t sizeOfStruct = scheme->getSizeOfDatastruct();
    byte* dbg_f = (byte*)dbg_field;
    byte* dbg_lr = (byte*)dbg_lr_halo;
    byte* dbg_tb = (byte*)dbg_tb_halo;
    byte* dbg_lrtb = (byte*)dbg_lrtb_halo;
    byte* cell = nullptr;
    byte* lrtb = nullptr;

    for(size_t i = 0; i < lN_Y; ++i) {
        for(size_t s = 0; s < sizeOfStruct; ++s) {
            dbg_lr[i * sizeOfStruct + s] = dbg_f[i * lN_X * sizeOfStruct + s];
            dbg_lr[(lN_Y + i) * sizeOfStruct + s] = dbg_f[((i + 1) * lN_X - 1) * sizeOfStruct + s];
        }
    }

    for(size_t i = 0; i < lN_X; ++i) {
        for(size_t s = 0; s < sizeOfStruct; ++s) {
            dbg_tb[i * sizeOfStruct + s] = dbg_f[i * sizeOfStruct + s];
            dbg_tb[(lN_X + i) * sizeOfStruct + s] = dbg_f[((lN_Y - 1) * lN_X + i) * sizeOfStruct + s];
        }
    }

    for(size_t i = 0; i < lrtb_size; ++i) {
        if(i == 0) { // CU_LEFT_TOP_BORDER
            cell = dbg_f;
            lrtb = dbg_lrtb;
        } else if(i == 1) { // CU_RIGHT_TOP_BORDER
            cell = &dbg_f[(lN_X - 1) * sizeOfStruct];
            lrtb = dbg_lrtb + sizeOfStruct;
        } else if(i == 2) { // CU_LEFT_BOTTOM_BORDER
            cell = &dbg_f[(lN_Y - 1) * lN_X * sizeOfStruct];
            lrtb = dbg_lrtb + 2 * sizeOfStruct;
        } else { // i == 3  // CU_RIGHT_BOTTOM_BORDER
            cell = &dbg_f[(lN_Y * lN_X - 1) * sizeOfStruct];
            lrtb = dbg_lrtb + 3 * sizeOfStruct;
        }
        for(size_t s = 0; s < sizeOfStruct; ++s) {
            lrtb[s] = cell[s];
        }
    }
    return GPU_SUCCESS;
}
#endif
