/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   MPI_Node.hpp
 * Author: eugene
 *
 * Created on November 1, 2017, 1:13 PM
 */

#ifndef MPI_NODE_HPP
#define MPI_NODE_HPP

#include "ComputationalModel.hpp"

class MPI_Node {
public:
    MPI_Node();
    MPI_Node(const MPI_Node& orig);
    virtual ~MPI_Node();
public:
    void loadXMLParserLib();
    void initEnvironment();
    virtual void runNode() = 0;
    
public:
    ComputationalModel* model;
};

#endif /* MPI_NODE_HPP */

