/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   LatticeBoltzmannModel.hpp
 * Author: eugene
 *
 * Created on November 1, 2017, 2:20 PM
 */

#ifndef LATTICEBOLTZMANNMODEL_HPP
#define LATTICEBOLTZMANNMODEL_HPP

#include "ComputationalModel.hpp"
#include <cstdlib>

class LatticeBoltzmannModel : public ComputationalModel {
public:
    LatticeBoltzmannModel();
    virtual ~LatticeBoltzmannModel();
    
public:
    virtual void createMpiStructType();
    virtual void* getField();
    
protected:
    
};

#endif /* LATTICEBOLTZMANNMODEL_HPP */

