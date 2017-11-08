/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   CPUTestComputationalModel.h
 * Author: eugene
 *
 * Created on November 8, 2017, 10:47 AM
 */

#ifndef CPUTESTCOMPUTATIONALMODEL_H
#define CPUTESTCOMPUTATIONALMODEL_H

#include "ComputationalModel.hpp"
#include "cell.h"

class CPUTestComputationalModel : public ComputationalModel {
public:
    CPUTestComputationalModel(const char* compModel, const char* gridModel);
    virtual ~CPUTestComputationalModel();

public:
    virtual void createMpiStructType();
    virtual void initializeField();
    virtual void* getTmpCPUFieldStoragePtr();
    virtual void updateGlobalField(size_t mpi_node_x, size_t mpi_node_y);
    virtual void prepareSubfield(size_t mpi_node_x = 0, size_t mpi_node_y = 0);
    virtual void loadSubFieldToGPU();
    virtual void* getField();
    virtual void gpuSync();
    virtual void performSimulationStep();
    virtual void updateHaloBorderElements();
    virtual void prepareHaloElements();
    virtual void* getCPUHaloPtr(size_t border_type);
    virtual void* getTmpCPUHaloPtr(size_t border_type);
    virtual void setStopMarker();
    virtual bool checkStopMarker();
    
protected:
    Cell* field;
    Cell* tmpCPUField;
    Cell* lr_halo;
    Cell* tb_halo;
    Cell* rcv_lr_halo;
    Cell* rcv_tb_halo;
};

#endif /* CPUTESTCOMPUTATIONALMODEL_H */

