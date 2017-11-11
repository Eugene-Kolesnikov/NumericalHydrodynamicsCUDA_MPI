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

#include "CPUComputationalModel.hpp"
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

void CPUComputationalModel::initializeField() 
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

void CPUComputationalModel::updateGlobalField(size_t mpi_node_x, size_t mpi_node_y) 
{
    if(nodeType != NODE_TYPE::SERVER_NODE)
        throw std::runtime_error("CPUComputationalModel::updateGlobalField: "
                "This function should not be called by a Computational Node");
    memcpyField(mpi_node_x, mpi_node_y, TmpCPUFieldToField);
}

void CPUComputationalModel::prepareSubfield(size_t mpi_node_x, size_t mpi_node_y) 
{
    if(nodeType == NODE_TYPE::COMPUTATIONAL_NODE) {
        // nothing yet
    } else {
        memcpyField(mpi_node_x, mpi_node_y, FieldToTmpCPUField);
    }
}

void CPUComputationalModel::loadSubFieldToGPU() 
{
    // nothing yet
}

void CPUComputationalModel::gpuSync() 
{
    // nothing yet
}

void CPUComputationalModel::performSimulationStep() 
{ // shift from right to the left
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUComputationalModel::performSimulationStep: "
                "This function should not be called by the Server Node");
    // for now use the CPU field
    scheme->performSimulationStep(tmpCPUField, lr_halo, tb_halo, lN_X, lN_Y);
}

void CPUComputationalModel::updateHaloBorderElements() 
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUComputationalModel::updateHaloBorderElements: "
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

void CPUComputationalModel::prepareHaloElements() 
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUComputationalModel::prepareHaloElements: "
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

void CPUComputationalModel::memcpyField(size_t mpi_node_x, size_t mpi_node_y, 
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
