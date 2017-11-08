/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "computationalModel_interface.h"
#include "ComputationalModel.hpp"
#include "CPUTestComputationalModel.hpp"

void* createComputationalModel(const char* compModel, const char* gridModel)
{
    ComputationalModel* model = new CPUTestComputationalModel(compModel, gridModel);
    return (void*) model;
}