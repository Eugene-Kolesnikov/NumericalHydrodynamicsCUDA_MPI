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
public:
    CPUComputationalModel(const char* compModel, const char* gridModel);
    virtual ~CPUComputationalModel();

public:
    virtual ErrorStatus initializeEnvironment();
    virtual ErrorStatus updateGlobalField(size_t mpi_node_x, size_t mpi_node_y);
    virtual ErrorStatus prepareSubfield(size_t mpi_node_x = 0, size_t mpi_node_y = 0);
    virtual ErrorStatus loadSubFieldToGPU();
    virtual ErrorStatus gpuSync();
    virtual ErrorStatus performSimulationStep();
    virtual ErrorStatus updateHaloBorderElements(size_t mpi_node_x, size_t mpi_node_y);
    virtual ErrorStatus prepareHaloElements();
    virtual ErrorStatus deinitModel();
};

#endif /* CPUCOMPUTATIONALMODEL_H */
