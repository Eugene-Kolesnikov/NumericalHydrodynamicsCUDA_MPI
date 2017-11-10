/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   CPUTestComputationalModel.cpp
 * Author: eugene
 * 
 * Created on November 8, 2017, 10:47 AM
 */

#include "CPUTestComputationalModel.hpp"
#include <typeinfo> // operator typeid
#include <exception>
#include <vector>
#include <cstdlib> //drand48
#include <mpi.h>

/**
 * byte type is actually a char type. The reason to use it is that all other 
 * types like float, double, long double can be represented as a sequence of 
 * chars, which means that size (memory) of each type is a multiple of the char size.
 * It is useful to use byte since we don't know which one of the types (float, double,
 * long double) will be used in the program.
 */
typedef char byte;

CPUTestComputationalModel::CPUTestComputationalModel(const char* compModel, const char* gridModel):
    ComputationalModel(compModel, gridModel)
{
    field = nullptr;
    tmpCPUField = nullptr;
    lr_halo = nullptr;
    tb_halo = nullptr;
    lrtb_halo = nullptr;
    rcv_lr_halo = nullptr;
    rcv_tb_halo = nullptr;
    rcv_lrtb_halo = nullptr;
}

CPUTestComputationalModel::~CPUTestComputationalModel() 
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
}

void CPUTestComputationalModel::createMpiStructType(logging::FileLogger& Log) 
{
    const std::type_info& data_typeid = scheme->getDataTypeid();
    size_t size_of_datatype = scheme->getSizeOfDatatype();
    int mpi_err_status, resultlen;
    char err_buffer[MPI_MAX_ERROR_STRING];
    MPI_Datatype MPI_DATA_TYPE;
    if(data_typeid == typeid(float)) {
        MPI_DATA_TYPE = MPI_FLOAT;
    } else if(data_typeid == typeid(double)) {
        MPI_DATA_TYPE = MPI_DOUBLE;
    } else if(data_typeid == typeid(long double)) {
        MPI_DATA_TYPE = MPI_LONG_DOUBLE;
    } else {
        throw std::runtime_error("CPUTestComputationalModel::createMpiStructType: "
                "Wrong STRUCT_DATA_TYPE!");
    }
    size_t nitems = scheme->getNumberOfElements();
    int* blocklengths = new int[nitems];
    MPI_Datatype* types = new MPI_Datatype[nitems];
    MPI_Aint* offsets = new MPI_Aint[nitems];
    for(size_t i = 0; i < nitems; ++i) {
        blocklengths[i] = 1;
        types[i] = MPI_DATA_TYPE;
        offsets[i] = i * size_of_datatype;
    }
    mpi_err_status = MPI_Type_create_struct(nitems, blocklengths, offsets, 
            types, &MPI_CellType);
    delete[] blocklengths;
    delete[] types;
    delete[] offsets;
    if(mpi_err_status != MPI_SUCCESS) {
        MPI_Error_string(mpi_err_status, err_buffer, &resultlen);
        throw std::runtime_error(err_buffer);
    }
    mpi_err_status = MPI_Type_commit(&MPI_CellType);
    if(mpi_err_status != MPI_SUCCESS) {
        MPI_Error_string(mpi_err_status, err_buffer, &resultlen);
        throw std::runtime_error(err_buffer);
    }
    Log << "MPI structure has been successfully created";
}

void CPUTestComputationalModel::initializeField() 
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
        scheme->initField(field, N_X, N_Y);
    }
}

void* CPUTestComputationalModel::getTmpCPUFieldStoragePtr() 
{
    return tmpCPUField;
}

void CPUTestComputationalModel::updateGlobalField(size_t mpi_node_x, size_t mpi_node_y) 
{
    if(nodeType != NODE_TYPE::SERVER_NODE)
        throw std::runtime_error("CPUTestComputationalModel::updateGlobalField: "
                "This function should not be called by a Computational Node");
    memcpyField(mpi_node_x, mpi_node_y, TmpCPUFieldToField);
}

void CPUTestComputationalModel::prepareSubfield(size_t mpi_node_x, size_t mpi_node_y) 
{
    if(nodeType == NODE_TYPE::COMPUTATIONAL_NODE) {
        // nothing yet
    } else {
        memcpyField(mpi_node_x, mpi_node_y, FieldToTmpCPUField);
    }
}

void CPUTestComputationalModel::loadSubFieldToGPU() 
{
    // nothing yet
}

void* CPUTestComputationalModel::getField() 
{
    return field;
}

void CPUTestComputationalModel::gpuSync() 
{
    // nothing yet
}

void CPUTestComputationalModel::performSimulationStep() 
{ // shift from right to the left
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUTestComputationalModel::performSimulationStep: "
                "This function should not be called by the Server Node");
    // for now use the CPU field
    scheme->performSimulationStep(tmpCPUField, lr_halo, tb_halo, lN_X, lN_Y);
}

void CPUTestComputationalModel::updateHaloBorderElements() 
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUTestComputationalModel::updateHaloBorderElements: "
                "This function should not be called by the Server Node");
    size_t shift, shift_item, global;
    size_t size_of_datatype = scheme->getSizeOfDatatype();
    size_t nitems = scheme->getNumberOfElements();
    // update lr
    byte* lr_haloPtr = (byte*)lr_halo;
    byte* rcv_lr_halodPtr = (byte*)rcv_lr_halo;
    for(size_t y = 0; y < 2*lN_Y; ++y) {
        /// Go through all elements of lr_halo array
        /** Calculate the shift using the fact that structure consists
         * of nitems amount of elements which are size_of_datatype amount
         * of bytes each. */
        shift = y * nitems * size_of_datatype;
        for(size_t i = 0; i < nitems; ++i) {
            /// Go through all elements of the Cell
            /** Add shifts for the elements inside the Cell structure */
            shift_item = shift + i * size_of_datatype;
            for(size_t s = 0; s < size_of_datatype; ++s) {
                /// Go through all bytes of the STRUCT_DATA_TYPE
                /** Add shifts for the bytes of the STRUCT_DATA_TYPE */
                global = shift_item + s;
                /// Update the byte
                lr_haloPtr[global] = rcv_lr_halodPtr[global];
            }
        }
    }
    // update tb
    byte* tb_haloPtr = (byte*)tb_halo;
    byte* rcv_tb_halodPtr = (byte*)rcv_tb_halo;
    for(size_t x = 0; x < 2*lN_X; ++x) {
        /// Go through all elements of lr_halo array
        /** Calculate the shift using the fact that structure consists
         * of nitems amount of elements which are size_of_datatype amount
         * of bytes each. */
        shift = x * nitems * size_of_datatype;
        for(size_t i = 0; i < nitems; ++i) {
            /// Go through all elements of the Cell
            /** Add shifts for the elements inside the Cell structure */
            shift_item = shift + i * size_of_datatype;
            for(size_t s = 0; s < size_of_datatype; ++s) {
                /// Go through all bytes of the STRUCT_DATA_TYPE
                /** Add shifts for the bytes of the STRUCT_DATA_TYPE */
                global = shift_item + s;
                /// Update the byte
                tb_haloPtr[global] = rcv_tb_halodPtr[global];
            }
        }
    }
}

void CPUTestComputationalModel::prepareHaloElements() 
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUTestComputationalModel::prepareHaloElements: "
                "This function should not be called by the Server Node");
    size_t halo_shift, halo_shift_item, halo_global;
    size_t halo_shift1, halo_shift_item1, halo_global1;
    size_t field_shift, field_shift_item, field_global;
    size_t field_shift1, field_shift_item1, field_global1;
    size_t size_of_datatype = scheme->getSizeOfDatatype();
    size_t nitems = scheme->getNumberOfElements();
    byte* tmpCPUFieldPtr = (byte*)tmpCPUField;
    // update lr
    byte* lr_haloPtr = (byte*)lr_halo;
    for(size_t y = 0; y < lN_Y; ++y) {
        /// Go through all elements of a column of the subfield
        /** Calculate the shift using the fact that structure consists
         * of nitems amount of elements which are size_of_datatype amount
         * of bytes each. */
        halo_shift = y * nitems * size_of_datatype;
        halo_shift1 = (y + lN_Y) * nitems * size_of_datatype;
        field_shift = (y * lN_X) * nitems * size_of_datatype;
        field_shift1 = ((y+1) * lN_X - 1) * nitems * size_of_datatype;
        for(size_t i = 0; i < nitems; ++i) {
            /// Go through all elements of the Cell
            /** Add shifts for the elements inside the Cell structure */
            halo_shift_item = halo_shift + i * size_of_datatype;
            halo_shift_item1 = halo_shift1 + i * size_of_datatype;
            field_shift_item = field_shift + i * size_of_datatype;
            field_shift_item1 = field_shift1 + i * size_of_datatype;
            for(size_t s = 0; s < size_of_datatype; ++s) {
                /// Go through all bytes of the STRUCT_DATA_TYPE
                /** Add shifts for the bytes of the STRUCT_DATA_TYPE */
                halo_global = halo_shift_item + s;
                halo_global1 = halo_shift_item1 + s;
                field_global = field_shift_item + s;
                field_global1 = field_shift_item1 + s;
                /// Update the byte
                lr_haloPtr[halo_global] = tmpCPUFieldPtr[field_global];
                lr_haloPtr[halo_global1] = tmpCPUFieldPtr[field_global1];
            }
        }
    }
    // update lr
    byte* tb_haloPtr = (byte*)tb_halo;
    for(size_t x = 0; x < lN_X; ++x) {
        /// Go through all elements of a column of the subfield
        /** Calculate the shift using the fact that structure consists
         * of nitems amount of elements which are size_of_datatype amount
         * of bytes each. */
        halo_shift = x * nitems * size_of_datatype;
        halo_shift1 = (x + lN_X) * nitems * size_of_datatype;
        field_shift = x * nitems * size_of_datatype;
        field_shift1 = ((lN_Y-1) * lN_X + x) * nitems * size_of_datatype;
        for(size_t i = 0; i < nitems; ++i) {
            /// Go through all elements of the Cell
            /** Add shifts for the elements inside the Cell structure */
            halo_shift_item = halo_shift + i * size_of_datatype;
            halo_shift_item1 = halo_shift1 + i * size_of_datatype;
            field_shift_item = field_shift + i * size_of_datatype;
            field_shift_item1 = field_shift1 + i * size_of_datatype;
            for(size_t s = 0; s < size_of_datatype; ++s) {
                /// Go through all bytes of the STRUCT_DATA_TYPE
                /** Add shifts for the bytes of the STRUCT_DATA_TYPE */
                halo_global = halo_shift_item + s;
                halo_global1 = halo_shift_item1 + s;
                field_global = field_shift_item + s;
                field_global1 = field_shift_item1 + s;
                /// Update the byte
                tb_haloPtr[halo_global] = tmpCPUFieldPtr[field_global];
                tb_haloPtr[halo_global1] = tmpCPUFieldPtr[field_global1];
            }
        }
    }
    // update lrtb
    byte* lrtb_haloPtr = (byte*)lrtb_halo;
    size_t global_field_shifts[4] = 
        {0, lN_X - 1, (lN_Y-1) * lN_X, lN_Y * lN_X - 1};
    for(size_t border = 0; border < 4; ++border) {
        halo_shift = border * nitems * size_of_datatype;
        field_shift = global_field_shifts[border] * nitems * size_of_datatype;
        for(size_t i = 0; i < nitems; ++i) {
            /// Go through all elements of the Cell
            /** Add shifts for the elements inside the Cell structure */
            halo_shift_item = halo_shift + i * size_of_datatype;
            field_shift_item = field_shift + i * size_of_datatype;
            for(size_t s = 0; s < size_of_datatype; ++s) {
                /// Go through all bytes of the STRUCT_DATA_TYPE
                /** Add shifts for the bytes of the STRUCT_DATA_TYPE */
                halo_global = halo_shift_item + s;
                field_global = field_shift_item + s;
                /// Update the byte
                lrtb_haloPtr[halo_global] = tmpCPUFieldPtr[field_global];
            }
        }
    }
}

void* CPUTestComputationalModel::getCPUHaloPtr(size_t border_type) 
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUTestComputationalModel::getCPUHaloPtr: "
                "This function should not be called by the Server Node");
    size_t size_of_datatype = scheme->getSizeOfDatatype();
    size_t nitems = scheme->getNumberOfElements();
    if(border_type == LEFT_BORDER)
        return lr_halo;
    else if(border_type == RIGHT_BORDER) {
        byte* lr_haloPtr = (byte*)lr_halo;
        byte* r_haloPtr = lr_haloPtr + lN_Y * nitems * size_of_datatype;
        return (void*)r_haloPtr;
    } else if(border_type == TOP_BORDER)
        return tb_halo;
    else if(border_type == BOTTOM_BORDER) {
        byte* tb_haloPtr = (byte*)tb_halo;
        byte* b_haloPtr = tb_haloPtr + lN_X * nitems * size_of_datatype;
        return (void*)b_haloPtr;
    } else
        throw std::runtime_error("CPUTestComputationalModel::getCPUHaloPtr: "
                "Wrong border_type");
}

void* CPUTestComputationalModel::getCPUDiagHaloPtr(size_t border_type)
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUTestComputationalModel::getCPUDiagHaloPtr: "
                "This function should not be called by the Server Node");
    size_t size_of_datatype = scheme->getSizeOfDatatype();
    size_t nitems = scheme->getNumberOfElements();
    if(border_type != LEFT_TOP_BORDER && border_type != RIGHT_TOP_BORDER && 
            border_type != LEFT_BOTTOM_BORDER && border_type != RIGHT_BOTTOM_BORDER)
        throw std::runtime_error("CPUTestComputationalModel::getCPUDiagHaloPtr: "
                "Wrong border_type");
    byte* lrtb_haloPtr = (byte*)lrtb_halo;
    byte* haloPtr = lrtb_haloPtr + border_type * nitems * size_of_datatype;
    return (void*)haloPtr;
}

void* CPUTestComputationalModel::getTmpCPUHaloPtr(size_t border_type) 
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUTestComputationalModel::getTmpCPUHaloPtr: "
                "This function should not be called by the Server Node");
    size_t size_of_datatype = scheme->getSizeOfDatatype();
    size_t nitems = scheme->getNumberOfElements();
    if(border_type == LEFT_BORDER)
        return rcv_lr_halo;
    else if(border_type == RIGHT_BORDER) {
        byte* rcv_lr_haloPtr = (byte*)rcv_lr_halo;
        byte* rcv_r_haloPtr = rcv_lr_haloPtr + lN_Y * nitems * size_of_datatype;
        return (void*)rcv_r_haloPtr;
    } else if(border_type == TOP_BORDER)
        return rcv_tb_halo;
    else if(border_type == BOTTOM_BORDER) {
        byte* rcv_tb_haloPtr = (byte*)rcv_tb_halo;
        byte* rcv_b_haloPtr = rcv_tb_haloPtr + lN_X * nitems * size_of_datatype;
        return (void*)rcv_b_haloPtr;
    } else
        throw std::runtime_error("CPUTestComputationalModel::getTmpCPUHaloPtr: "
                "Wrong border_type");
}

void* CPUTestComputationalModel::getTmpCPUDiagHaloPtr(size_t border_type)
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUTestComputationalModel::getTmpCPUDiagHaloPtr: "
                "This function should not be called by the Server Node");
    size_t size_of_datatype = scheme->getSizeOfDatatype();
    size_t nitems = scheme->getNumberOfElements();
    if(border_type != LEFT_TOP_BORDER && border_type != RIGHT_TOP_BORDER && 
            border_type != LEFT_BOTTOM_BORDER && border_type != RIGHT_BOTTOM_BORDER)
        throw std::runtime_error("CPUTestComputationalModel::getTmpCPUDiagHaloPtr: "
                "Wrong border_type");
    byte* rcv_lrtb_haloPtr = (byte*)rcv_lrtb_halo;
    byte* rcv_haloPtr = rcv_lrtb_haloPtr + border_type * nitems * size_of_datatype;
    return (void*)rcv_haloPtr;
}

void CPUTestComputationalModel::setStopMarker() 
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUTestComputationalModel::setStopMarker: "
                "This function should not be called by the Server Node");
    size_t size_of_datatype = scheme->getSizeOfDatatype();
    byte* markerValue = (byte*)scheme->getMarkerValue();
    byte* tmpCPUFieldPtr = (byte*)tmpCPUField;
    for(size_t s = 0; s < size_of_datatype; ++s) {
        /// Go through all bytes of the STRUCT_DATA_TYPE
        /// Update the byte
        tmpCPUFieldPtr[s] = markerValue[s];
    }
}

bool CPUTestComputationalModel::checkStopMarker()
{
    if(nodeType != NODE_TYPE::SERVER_NODE)
        throw std::runtime_error("CPUTestComputationalModel::checkStopMarker: "
                "This function should not be called by a Computational Node");
    size_t size_of_datatype = scheme->getSizeOfDatatype();
    byte* markerValue = (byte*)scheme->getMarkerValue();
    byte* fieldPtr = (byte*)field;
    for(size_t s = 0; s < size_of_datatype; ++s) {
        /// Go through all bytes of the STRUCT_DATA_TYPE
        /// Update the byte
        if(fieldPtr[s] != markerValue[s])
            return false;
    }
    return true;
}

void CPUTestComputationalModel::memcpyField(size_t mpi_node_x, size_t mpi_node_y, 
        TypeMemCpy cpyType)
{
    byte* fieldPtr = (byte*)field;
    byte* tmpCPUFieldPtr = (byte*)tmpCPUField;
    size_t size_of_datatype = scheme->getSizeOfDatatype();
    size_t nitems = scheme->getNumberOfElements();
    size_t global, globalTmp, x0, y0, x1, y1;
    size_t global_shift, global_shiftTmp;
    size_t global_shift_item, global_shiftTmp_item;
    // calculate the shift for the particular subfield
    x0 = static_cast<size_t>(N_X / MPI_NODES_X * mpi_node_x);
    y0 = static_cast<size_t>(N_Y / MPI_NODES_Y * mpi_node_y);
    for(size_t x = 0; x < lN_X; ++x) {
        /// Go through all x elements of the subfield
        /** Add the shift to the x-component since the subfield can be located
         * somewhere inside the global field */
        x1 = x0 + x;
        for(size_t y = 0; y < lN_Y; ++y) {
            /// Go through all y elements of the subfield
            /** Add the shift to the y-component since the subfield can be located
             * somewhere inside the global field */
            y1 = y0 + y;
            /** Calculate the global shifts for the 'global field' and 'subfield'
             * using the fact that structure consists of nitems amount of 
             * elements which are size_of_datatype amount of bytes each. */
            global_shift = (y1 * N_X + x1) * nitems * size_of_datatype;
            global_shiftTmp = (y * lN_X + x) * nitems * size_of_datatype;
            for(size_t i = 0; i < nitems; ++i) {
                /// Go through all elements of the Cell
                /** Add shifts for the elements inside the Cell structure */
                global_shift_item = global_shift + i * size_of_datatype;
                global_shiftTmp_item = global_shiftTmp + i * size_of_datatype;
                for(size_t s = 0; s < size_of_datatype; ++s) {
                    /// Go through all bytes of the STRUCT_DATA_TYPE
                    /** Add shifts for the bytes of the STRUCT_DATA_TYPE */
                    global = global_shift_item + s;
                    globalTmp = global_shiftTmp_item + s;
                    /// Update the byte
                    if(cpyType == TmpCPUFieldToField)
                        fieldPtr[global] = tmpCPUFieldPtr[globalTmp];
                    else // FieldToTmpCPUField
                        tmpCPUFieldPtr[globalTmp] = fieldPtr[global];
                }
            }
        }
    }
}
