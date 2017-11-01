/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   ComputationalModel.hpp
 * Author: eugene
 *
 * Created on November 1, 2017, 12:51 PM
 */

#ifndef COMPUTATIONALMODEL_HPP
#define COMPUTATIONALMODEL_HPP

class ComputationalModel {
public:
    ComputationalModel();
    ComputationalModel(const ComputationalModel& orig);
    virtual ~ComputationalModel();
    
public:
    virtual void createMpiStructType() = 0;

public:
    static MPI_Datatype MPI_CellType;
};

#endif /* COMPUTATIONALMODEL_HPP */

