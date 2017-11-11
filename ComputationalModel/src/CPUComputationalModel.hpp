/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   CPUComputationalModel.h
 * Author: eugene
 *
 * Created on November 8, 2017, 10:47 AM
 */

#ifndef CPUCOMPUTATIONALMODEL_H
#define CPUCOMPUTATIONALMODEL_H

#include "ComputationalModel.hpp"

class CPUComputationalModel : public ComputationalModel {
    enum TypeMemCpy {TmpCPUFieldToField, FieldToTmpCPUField};
public:
    CPUComputationalModel(const char* compModel, const char* gridModel);
    virtual ~CPUComputationalModel();

public:
    virtual void initializeField();
    virtual void updateGlobalField(size_t mpi_node_x, size_t mpi_node_y);
    virtual void prepareSubfield(size_t mpi_node_x = 0, size_t mpi_node_y = 0);
    virtual void loadSubFieldToGPU();
    virtual void gpuSync();
    virtual void performSimulationStep();
    virtual void updateHaloBorderElements();
    virtual void prepareHaloElements();
    
protected:
    void memcpyField(size_t mpi_node_x, size_t mpi_node_y, TypeMemCpy cpyType);
};

#endif /* CPUCOMPUTATIONALMODEL_H */

