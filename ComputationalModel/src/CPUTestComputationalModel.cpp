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
#include "cell.h"
#include <typeinfo> // operator typeid
#include <exception>
#include <vector>
#include <cstdlib> //drand48
#include <mpi.h>


CPUTestComputationalModel::CPUTestComputationalModel(const char* compModel, const char* gridModel):
    ComputationalModel(compModel, gridModel)
{
    field = nullptr;
    tmpCPUField = nullptr;
    lr_halo = nullptr;
    tb_halo = nullptr;
    rcv_lr_halo = nullptr;
    rcv_tb_halo = nullptr;
}

CPUTestComputationalModel::~CPUTestComputationalModel() 
{
    if(field != nullptr)
        delete[] field;
    if(lr_halo != nullptr)
        delete[] lr_halo;
    if(tb_halo != nullptr)
        delete[] tb_halo;
    if(rcv_lr_halo != nullptr)
        delete[] rcv_lr_halo;
    if(rcv_tb_halo != nullptr)
        delete[] rcv_tb_halo;
    if(tmpCPUField != nullptr)
        delete[] tmpCPUField;
}

void CPUTestComputationalModel::createMpiStructType() 
{
    int mpi_err_status, resultlen;
    char err_buffer[MPI_MAX_ERROR_STRING];
    MPI_Datatype MPI_DATA_TYPE;
    if(typeid(STRUCT_DATA_TYPE) == typeid(float)) {
        MPI_DATA_TYPE = MPI_FLOAT;
    } else if(typeid(STRUCT_DATA_TYPE) == typeid(double)) {
        MPI_DATA_TYPE = MPI_DOUBLE;
    } else if(typeid(STRUCT_DATA_TYPE) == typeid(long double)) {
        MPI_DATA_TYPE = MPI_LONG_DOUBLE;
    } else {
        throw std::runtime_error("CPUTestComputationalModel::createMpiStructType: Wrong STRUCT_DATA_TYPE!");
    }
    size_t nitems = static_cast<size_t>(sizeof(Cell) / sizeof(STRUCT_DATA_TYPE));
    int blocklengths[MAX_CELL_ARG];
    MPI_Datatype types[MAX_CELL_ARG];
    MPI_Aint offsets[MAX_CELL_ARG];
    
    for(size_t i = 0; i < nitems; ++i) {
        blocklengths[i] = 1;
        types[i] = MPI_DATA_TYPE;
        offsets[i] = i * sizeof(STRUCT_DATA_TYPE);
    }
    mpi_err_status = MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_CellType);
    if(mpi_err_status != MPI_SUCCESS) {
        MPI_Error_string(mpi_err_status, err_buffer, &resultlen);
        throw std::runtime_error(err_buffer);
    }
    mpi_err_status = MPI_Type_commit(&MPI_CellType);
    if(mpi_err_status != MPI_SUCCESS) {
        MPI_Error_string(mpi_err_status, err_buffer, &resultlen);
        throw std::runtime_error(err_buffer);
    }
}

void CPUTestComputationalModel::initializeField() 
{
    tmpCPUField = new Cell[lN_X * lN_Y];
    if(nodeType == NODE_TYPE::COMPUTATIONAL_NODE) {
        lr_halo = new Cell[2*lN_Y];
        tb_halo = new Cell[2*lN_X];
        rcv_lr_halo = new Cell[2*lN_Y];
        rcv_tb_halo = new Cell[2*lN_X];
    } else {
        size_t global;
        field = new Cell[N_X * N_Y];
        for(size_t x = 0; x < N_X; ++x) {
            for(size_t y = 0; y < N_Y; ++y) {
                global = y * N_X + x;
                field[global].r = drand48();
                field[global].u = 0.0;
                field[global].v = 0.0;
                field[global].e = 0.0;
            }
        }
    }
}

void* CPUTestComputationalModel::getTmpCPUFieldStoragePtr() 
{
    return (void*) tmpCPUField;
}

void CPUTestComputationalModel::updateGlobalField(size_t mpi_node_x, size_t mpi_node_y) 
{
    if(nodeType != NODE_TYPE::SERVER_NODE)
        throw std::runtime_error("CPUTestComputationalModel::updateGlobalField: "
                "This function should not be called by a Computational Node");
    size_t global, globalTmp, x0, y0, x1, y1;
    x0 = static_cast<size_t>(N_X / MPI_NODES_X * mpi_node_x);
    y0 = static_cast<size_t>(N_Y / MPI_NODES_Y * mpi_node_y);
    for(size_t x = 0; x < lN_X; ++x) {
        for(size_t y = 0; y < lN_Y; ++y) {
            x1 = x0 + x;
            y1 = y0 + y;
            global = y1 * N_X + x1;
            globalTmp = y * lN_X + x;
            field[global].r = tmpCPUField[globalTmp].r;
            field[global].u = tmpCPUField[globalTmp].u;
            field[global].v = tmpCPUField[globalTmp].v;
            field[global].e = tmpCPUField[globalTmp].e;
        }
    }
}

void CPUTestComputationalModel::prepareSubfield(size_t mpi_node_x, size_t mpi_node_y) 
{
    if(nodeType == NODE_TYPE::COMPUTATIONAL_NODE) {
        // nothing
    } else {
        size_t global, globalTmp, x0, y0, x1, y1;
        x0 = static_cast<size_t>(N_X / MPI_NODES_X * mpi_node_x);
        y0 = static_cast<size_t>(N_Y / MPI_NODES_Y * mpi_node_y);
        for(size_t x = 0; x < lN_X; ++x) {
            for(size_t y = 0; y < lN_Y; ++y) {
                x1 = x0 + x;
                y1 = y0 + y;
                global = y1 * N_X + x1;
                globalTmp = y * lN_X + x;
                tmpCPUField[globalTmp].r = field[global].r;
                tmpCPUField[globalTmp].u = field[global].u;
                tmpCPUField[globalTmp].v = field[global].v;
                tmpCPUField[globalTmp].e = field[global].e;
            }
        }
    }
}

void CPUTestComputationalModel::loadSubFieldToGPU() 
{
    // nothing
}

void* CPUTestComputationalModel::getField() 
{
    return (void*)field;
}

void CPUTestComputationalModel::gpuSync() 
{
    // nothing
}

void CPUTestComputationalModel::performSimulationStep() 
{ // shift from right to the left
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUTestComputationalModel::performSimulationStep: "
                "This function should not be called by the Server Node");
    size_t global, global1;
    Cell* tmpField = new Cell[lN_Y];
    for(size_t y = 0; y < lN_Y; ++y) {
        global = y * lN_X;
        tmpField[y].r = tmpCPUField[global].r;
        tmpField[y].u = tmpCPUField[global].u;
        tmpField[y].v = tmpCPUField[global].v;
        tmpField[y].e = tmpCPUField[global].e;
    }
    for(size_t x = 1; x < lN_X; ++x) {
        for(size_t y = 0; y < lN_Y; ++y) {
            global = y * lN_X + x;
            global1 = y * lN_X + x - 1;
            tmpCPUField[global1].r = tmpCPUField[global].r;
            tmpCPUField[global1].u = tmpCPUField[global].u;
            tmpCPUField[global1].v = tmpCPUField[global].v;
            tmpCPUField[global1].e = tmpCPUField[global].e;
        }
    }
    for(size_t y = 1; y <= lN_Y; ++y) {
        global = y * lN_X - 1;
        tmpCPUField[global].r = tmpField[y-1].r;
        tmpCPUField[global].u = tmpField[y-1].u;
        tmpCPUField[global].v = tmpField[y-1].v;
        tmpCPUField[global].e = tmpField[y-1].e;
    }
    delete[] tmpField;
}

void CPUTestComputationalModel::updateHaloBorderElements() 
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUTestComputationalModel::updateHaloBorderElements: "
                "This function should not be called by the Server Node");
    // update lr
    for(size_t y = 0; y < 2*lN_Y; ++y) {
        lr_halo[y].r = rcv_lr_halo[y].r;
        lr_halo[y].u = rcv_lr_halo[y].u;
        lr_halo[y].v = rcv_lr_halo[y].v;
        lr_halo[y].e = rcv_lr_halo[y].e;
    }
    // update tb
    for(size_t x = 0; x < 2*lN_X; ++x) {
        tb_halo[x].r = rcv_tb_halo[x].r;
        tb_halo[x].u = rcv_tb_halo[x].u;
        tb_halo[x].v = rcv_tb_halo[x].v;
        tb_halo[x].e = rcv_tb_halo[x].e;
    }
}

void CPUTestComputationalModel::prepareHaloElements() 
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUTestComputationalModel::prepareHaloElements: "
                "This function should not be called by the Server Node");
    size_t global, global1;
    for(size_t y = 0; y < lN_Y; ++y) {
        global = y * lN_X;
        global1 = (y+1) * lN_X - 1;
        lr_halo[y].r = tmpCPUField[global].r;
        lr_halo[y].u = tmpCPUField[global].u;
        lr_halo[y].v = tmpCPUField[global].v;
        lr_halo[y].e = tmpCPUField[global].e;
        lr_halo[y+lN_Y].r = tmpCPUField[global1].r;
        lr_halo[y+lN_Y].u = tmpCPUField[global1].u;
        lr_halo[y+lN_Y].v = tmpCPUField[global1].v;
        lr_halo[y+lN_Y].e = tmpCPUField[global1].e;
    }
    for(size_t x = 0; x < lN_X; ++x) {
        global = x;
        global1 = (lN_Y-1) * lN_X + x;
        tb_halo[x].r = tmpCPUField[global].r;
        tb_halo[x].u = tmpCPUField[global].u;
        tb_halo[x].v = tmpCPUField[global].v;
        tb_halo[x].e = tmpCPUField[global].e;
        tb_halo[x+lN_X].r = tmpCPUField[global1].r;
        tb_halo[x+lN_X].u = tmpCPUField[global1].u;
        tb_halo[x+lN_X].v = tmpCPUField[global1].v;
        tb_halo[x+lN_X].e = tmpCPUField[global1].e;
    }
}

void* CPUTestComputationalModel::getCPUHaloPtr(size_t border_type) 
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUTestComputationalModel::getCPUHaloPtr: "
                "This function should not be called by the Server Node");
    if(border_type == LEFT_BORDER)
        return (void*)lr_halo;
    else if(border_type == RIGHT_BORDER)
        return (void*)(lr_halo + lN_Y);
    else if(border_type == TOP_BORDER)
        return (void*)tb_halo;
    else if(border_type == BOTTOM_BORDER)
        return (void*)(tb_halo + lN_X);
    else
        throw std::runtime_error("CPUTestComputationalModel::getCPUHaloPtr: Wrong border_type");
}

void* CPUTestComputationalModel::getTmpCPUHaloPtr(size_t border_type) 
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUTestComputationalModel::getTmpCPUHaloPtr: "
                "This function should not be called by the Server Node");
    if(border_type == LEFT_BORDER)
        return (void*)rcv_lr_halo;
    else if(border_type == RIGHT_BORDER)
        return (void*)(rcv_lr_halo + lN_Y);
    else if(border_type == TOP_BORDER)
        return (void*)rcv_tb_halo;
    else if(border_type == BOTTOM_BORDER)
        return (void*)(rcv_tb_halo + lN_X);
    else
        throw std::runtime_error("CPUTestComputationalModel::getTmpCPUHaloPtr: Wrong border_type");
}

void CPUTestComputationalModel::setStopMarker() 
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("CPUTestComputationalModel::setStopMarker: "
                "This function should not be called by the Server Node");
    tmpCPUField[0].r = -1;
}

bool CPUTestComputationalModel::checkStopMarker()
{
    if(nodeType != NODE_TYPE::SERVER_NODE)
        throw std::runtime_error("CPUTestComputationalModel::checkStopMarker: "
                "This function should not be called by a Computational Node");
    return field[0].r == -1;
}

