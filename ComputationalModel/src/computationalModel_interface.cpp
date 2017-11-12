/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "computationalModel_interface.h"
#include "ComputationalModel.hpp"
#include "CPUComputationalModel.hpp"
#include "GPUComputationalModel.hpp"

void* createComputationalModel(const char* compModel, const char* gridModel)
{
    CPUComputationalModel* model = new CPUComputationalModel(compModel, gridModel);
    return (void*) model;
}