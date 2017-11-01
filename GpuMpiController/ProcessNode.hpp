/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   ProcessNode.hpp
 * Author: eugene
 *
 * Created on November 1, 2017, 12:49 PM
 */

#ifndef PROCESSNODE_HPP
#define PROCESSNODE_HPP

#include "MPI_Node.hpp"

class ProcessNode : public MPI_Node {
public:
    ProcessNode();
    ProcessNode(const ProcessNode& orig);
    virtual ~ProcessNode();
private:

};

#endif /* PROCESSNODE_HPP */

