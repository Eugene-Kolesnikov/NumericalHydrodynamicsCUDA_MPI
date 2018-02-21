/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   CPUComputationalModel.cpp
 * Author: eugene
 *
 * Created on November 8, 2017, 10:47 AM
 */

#include <ComputationalModel/include/CPU/CPUComputationalModel.hpp>
#include <typeinfo> // operator typeid
#include <exception>
#include <vector>
#include <cstdlib> //drand48

CPUComputationalModel::CPUComputationalModel(const char* compModel, const char* gridModel):
    ComputationalModel(compModel, gridModel)
{
}

CPUComputationalModel::~CPUComputationalModel()
{
}

ErrorStatus CPUComputationalModel::updateGlobalField(size_t mpi_node_x, size_t mpi_node_y)
{
    if(nodeType != NODE_TYPE::SERVER_NODE)
        throw std::runtime_error("CPUComputationalModel::updateGlobalField: "
                "This function should not be called by a Computational Node");
    memcpyField(mpi_node_x, mpi_node_y, TmpCPUFieldToField);
    return GPU_SUCCESS;
}

ErrorStatus CPUComputationalModel::initializeEnvironment()
{
    tmpCPUField = scheme->createField(lN_X, lN_Y);
    if(nodeType == NODE_TYPE::COMPUTATIONAL_NODE) {
        lr_halo = scheme->initHalos(2*lN_Y);
        tb_halo = scheme->initHalos(2*lN_X);
        lrtb_halo = scheme->initHalos(4);
        rcv_lr_halo = scheme->initHalos(2*lN_Y);
        rcv_tb_halo = scheme->initHalos(2*lN_X);
        rcv_lrtb_halo = scheme->initHalos(4);
    } else { // NODE_TYPE::SERVER_NODE
        field = scheme->createField(N_X, N_Y);
    }
    return GPU_SUCCESS;
}

ErrorStatus CPUComputationalModel::prepareSubfield(size_t mpi_node_x, size_t mpi_node_y)
{
    if(nodeType == NODE_TYPE::COMPUTATIONAL_NODE) {
        // No CPU<->GPU interaction in the CPU version
    } else {
        memcpyField(mpi_node_x, mpi_node_y, FieldToTmpCPUField);
    }
    return GPU_SUCCESS;
}

ErrorStatus CPUComputationalModel::loadSubFieldToGPU()
{
    // No CPU<->GPU interaction in the CPU version
    return GPU_SUCCESS;
}

ErrorStatus CPUComputationalModel::gpuSync()
{
    // No CPU<->GPU interaction in the CPU version
    return GPU_SUCCESS;
}

ErrorStatus CPUComputationalModel::performSimulationStep()
{ // shift from right to the left
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUComputationalModel::performSimulationStep: "
                "This function should not be called by the Server Node");
    // Use the CPU field in the CPU version
    scheme->performCPUSimulationStep(tmpCPUField, lr_halo, tb_halo, lrtb_halo, lN_X, lN_Y);
    return GPU_SUCCESS;
}

ErrorStatus CPUComputationalModel::updateHaloBorderElements(size_t mpi_node_x, size_t mpi_node_y)
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUComputationalModel::updateHaloBorderElements: "
                "This function should not be called by the Server Node");
    size_t shift, global;
    size_t sizeOfDataStruct = scheme->getSizeOfDatastruct();
    /// update lr
    byte* lr_haloPtr = (byte*)lr_halo;
    byte* rcv_lr_halodPtr = (byte*)rcv_lr_halo;
    for(size_t y = 0; y < 2*lN_Y; ++y) {
        /// Go through all elements of lr_halo array
        shift = y * sizeOfDataStruct;
        for(size_t s = 0; s < sizeOfDataStruct; ++s) {
            /// Go through all bytes of the Cell struct
            global = shift + s;
            /// Update the byte
            lr_haloPtr[global] = rcv_lr_halodPtr[global];
        }
    }
    /// update tb
    byte* tb_haloPtr = (byte*)tb_halo;
    byte* rcv_tb_halodPtr = (byte*)rcv_tb_halo;
    for(size_t x = 0; x < 2*lN_X; ++x) {
        /// Go through all elements of lr_halo array
        shift = x * sizeOfDataStruct;
        for(size_t s = 0; s < sizeOfDataStruct; ++s) {
            /// Go through all bytes of the Cell struct
            global = shift + s;
            /// Update the byte
            tb_haloPtr[global] = rcv_tb_halodPtr[global];
        }
    }
    /// update lrtb
    byte* lrtb_haloPtr = (byte*)lrtb_halo;
    byte* rcv_lrtb_halodPtr = (byte*)rcv_lrtb_halo;
    for(size_t i = 0; i < 4; ++i) {
        /// Go through all elements of lr_halo array
        shift = i * sizeOfDataStruct;
        for(size_t s = 0; s < sizeOfDataStruct; ++s) {
            /// Go through all bytes of the Cell struct
            global = shift + s;
            /// Update the byte
            lrtb_haloPtr[global] = rcv_lrtb_halodPtr[global];
        }
    }
    /// Update global borders
	if(mpi_node_x == 0) {
		/// Update global left border
		CM_HANDLE_GPUERROR(scheme->updateCPUGlobalBorders(tmpCPUField, lr_halo,
            tb_halo, lrtb_halo, lN_X, lN_Y, CU_LEFT_BORDER));
        CM_HANDLE_GPUERROR(scheme->updateCPUGlobalBorders(tmpCPUField, lr_halo,
            tb_halo, lrtb_halo, lN_X, lN_Y, CU_LEFT_TOP_BORDER));
        CM_HANDLE_GPUERROR(scheme->updateCPUGlobalBorders(tmpCPUField, lr_halo,
            tb_halo, lrtb_halo, lN_X, lN_Y, CU_LEFT_BOTTOM_BORDER));
	}
    if(mpi_node_x == (MPI_NODES_X - 1)) {
		/// Update global right border
		CM_HANDLE_GPUERROR(scheme->updateCPUGlobalBorders(tmpCPUField, lr_halo,
            tb_halo, lrtb_halo, lN_X, lN_Y, CU_RIGHT_BORDER));
        CM_HANDLE_GPUERROR(scheme->updateCPUGlobalBorders(tmpCPUField, lr_halo,
            tb_halo, lrtb_halo, lN_X, lN_Y, CU_RIGHT_TOP_BORDER));
        CM_HANDLE_GPUERROR(scheme->updateCPUGlobalBorders(tmpCPUField, lr_halo,
            tb_halo, lrtb_halo, lN_X, lN_Y, CU_RIGHT_BOTTOM_BORDER));
	}
	if(mpi_node_y == 0) {
		/// Update global top border
		CM_HANDLE_GPUERROR(scheme->updateCPUGlobalBorders(tmpCPUField, lr_halo,
            tb_halo, lrtb_halo, lN_X, lN_Y, CU_TOP_BORDER));
        CM_HANDLE_GPUERROR(scheme->updateCPUGlobalBorders(tmpCPUField, lr_halo,
            tb_halo, lrtb_halo, lN_X, lN_Y, CU_LEFT_TOP_BORDER));
        CM_HANDLE_GPUERROR(scheme->updateCPUGlobalBorders(tmpCPUField, lr_halo,
            tb_halo, lrtb_halo, lN_X, lN_Y, CU_RIGHT_TOP_BORDER));
	}
    if(mpi_node_y == (MPI_NODES_Y - 1)) {
		/// Update global bottom border
		CM_HANDLE_GPUERROR(scheme->updateCPUGlobalBorders(tmpCPUField, lr_halo,
            tb_halo, lrtb_halo, lN_X, lN_Y, CU_BOTTOM_BORDER));
        CM_HANDLE_GPUERROR(scheme->updateCPUGlobalBorders(tmpCPUField, lr_halo,
            tb_halo, lrtb_halo, lN_X, lN_Y, CU_LEFT_BOTTOM_BORDER));
        CM_HANDLE_GPUERROR(scheme->updateCPUGlobalBorders(tmpCPUField, lr_halo,
            tb_halo, lrtb_halo, lN_X, lN_Y, CU_RIGHT_BOTTOM_BORDER));
	}
    return GPU_SUCCESS;
}

ErrorStatus CPUComputationalModel::prepareHaloElements()
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUComputationalModel::prepareHaloElements: "
                "This function should not be called by the Server Node");
    size_t halo_shift, halo_global;
    size_t halo_shift1, halo_global1;
    size_t field_shift, field_global;
    size_t field_shift1, field_global1;
    size_t sizeOfDataStruct = scheme->getSizeOfDatastruct();
    byte* tmpCPUFieldPtr = (byte*)tmpCPUField;
    // update lr
    byte* lr_haloPtr = (byte*)lr_halo;
    for(size_t y = 0; y < lN_Y; ++y) {
        /// Go through all elements of a column of the subfield
        /** Calculate the shift using the fact that structure consists
         * of nitems amount of elements which are size_of_datatype amount
         * of bytes each. */
        halo_shift = y * sizeOfDataStruct;
        halo_shift1 = (y + lN_Y) * sizeOfDataStruct;
        field_shift = (y * lN_X) * sizeOfDataStruct;
        field_shift1 = ((y+1) * lN_X - 1) * sizeOfDataStruct;
        for(size_t s = 0; s < sizeOfDataStruct; ++s) {
            /// Go through all bytes of the STRUCT_DATA_TYPE
            /** Add shifts for the bytes of the STRUCT_DATA_TYPE */
            halo_global = halo_shift + s;
            halo_global1 = halo_shift1 + s;
            field_global = field_shift + s;
            field_global1 = field_shift1 + s;
            /// Update the byte
            lr_haloPtr[halo_global] = tmpCPUFieldPtr[field_global];
            lr_haloPtr[halo_global1] = tmpCPUFieldPtr[field_global1];
        }
    }
    // update tb
    byte* tb_haloPtr = (byte*)tb_halo;
    for(size_t x = 0; x < lN_X; ++x) {
        /// Go through all elements of a column of the subfield
        /** Calculate the shift using the fact that structure consists
         * of nitems amount of elements which are size_of_datatype amount
         * of bytes each. */
        halo_shift = x * sizeOfDataStruct;
        halo_shift1 = (x + lN_X) * sizeOfDataStruct;
        field_shift = x * sizeOfDataStruct;
        field_shift1 = ((lN_Y-1) * lN_X + x) * sizeOfDataStruct;
        for(size_t s = 0; s < sizeOfDataStruct; ++s) {
            /// Go through all bytes of the STRUCT_DATA_TYPE
            /** Add shifts for the bytes of the STRUCT_DATA_TYPE */
            halo_global = halo_shift + s;
            halo_global1 = halo_shift1 + s;
            field_global = field_shift + s;
            field_global1 = field_shift1 + s;
            /// Update the byte
            tb_haloPtr[halo_global] = tmpCPUFieldPtr[field_global];
            tb_haloPtr[halo_global1] = tmpCPUFieldPtr[field_global1];
        }
    }
    // update lrtb
    byte* lrtb_haloPtr = (byte*)lrtb_halo;
    size_t global_field_shifts[4] =
        {0, lN_X - 1, (lN_Y-1) * lN_X, lN_Y * lN_X - 1};
    for(size_t border = 0; border < 4; ++border) {
        halo_shift = border * sizeOfDataStruct;
        field_shift = global_field_shifts[border] * sizeOfDataStruct;
        for(size_t s = 0; s < sizeOfDataStruct; ++s) {
            /// Go through all bytes of the STRUCT_DATA_TYPE
            /** Add shifts for the bytes of the STRUCT_DATA_TYPE */
            halo_global = halo_shift + s;
            field_global = field_shift + s;
            /// Update the byte
            lrtb_haloPtr[halo_global] = tmpCPUFieldPtr[field_global];
        }
    }
    return GPU_SUCCESS;
}

ErrorStatus CPUComputationalModel::deinitModel()
{
    if(field != nullptr)
        delete[] (byte*)field;
    if(tmpCPUField != nullptr)
        delete[] (byte*)tmpCPUField;
    if(lr_halo != nullptr)
        delete[] (byte*)lr_halo;
    if(tb_halo != nullptr)
        delete[] (byte*)tb_halo;
    if(lrtb_halo != nullptr)
        delete[] (byte*)lrtb_halo;
    if(rcv_lr_halo != nullptr)
        delete[] (byte*)rcv_lr_halo;
    if(rcv_tb_halo != nullptr)
        delete[] (byte*)rcv_tb_halo;
    if(rcv_lrtb_halo != nullptr)
        delete[] (byte*)rcv_lrtb_halo;
    return GPU_SUCCESS;
}
