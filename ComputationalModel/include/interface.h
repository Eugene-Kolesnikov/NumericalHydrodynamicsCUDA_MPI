/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   computationalModel_interface.h
 * Author: eugene
 *
 * Created on November 4, 2017, 6:18 PM
 */

#ifndef COMPUTATIONALMODEL_INTERFACE_H
#define COMPUTATIONALMODEL_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

void* createComputationalModel(const char* compModel, const char* gridModel);


#ifdef __cplusplus
}
#endif

#endif /* COMPUTATIONALMODEL_INTERFACE_H */

