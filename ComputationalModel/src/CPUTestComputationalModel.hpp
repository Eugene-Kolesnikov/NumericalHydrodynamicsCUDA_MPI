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

class CPUTestComputationalModel : public ComputationalModel {
    enum TypeMemCpy {TmpCPUFieldToField, FieldToTmpCPUField};
public:
    CPUTestComputationalModel(const char* compModel, const char* gridModel);
    virtual ~CPUTestComputationalModel();

public:
    virtual void createMpiStructType(logging::FileLogger& Log);
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
    virtual void* getCPUDiagHaloPtr(size_t border_type);
    virtual void* getTmpCPUHaloPtr(size_t border_type);
    virtual void* getTmpCPUDiagHaloPtr(size_t border_type);
    virtual void setStopMarker();
    virtual bool checkStopMarker();
    
protected:
    void memcpyField(size_t mpi_node_x, size_t mpi_node_y, TypeMemCpy cpyType);
    
protected:
    void* field;
    void* tmpCPUField;
    void* lr_halo;
    void* tb_halo;
    void* lrtb_halo;
    void* rcv_lr_halo;
    void* rcv_tb_halo;
    void* rcv_lrtb_halo;
};

#endif /* CPUTESTCOMPUTATIONALMODEL_H */

