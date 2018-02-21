/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <ComputationalModel/include/interface.h>
#include <ComputationalModel/include/ComputationalModel.hpp>
#include <ComputationalModel/include/CPU/CPUComputationalModel.hpp>
#include <ComputationalModel/include/GPU/GPUComputationalModel.hpp>

void* createComputationalModel(const char* compModel, const char* gridModel)
{
    GPUComputationalModel* model = new GPUComputationalModel(compModel, gridModel);
    //CPUComputationalModel* model = new CPUComputationalModel(compModel, gridModel);
    return (void*) model;
}
