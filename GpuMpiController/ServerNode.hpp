/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   ServerNode.hpp
 * Author: eugene
 *
 * Created on November 1, 2017, 12:49 PM
 */

#ifndef SERVERNODE_HPP
#define SERVERNODE_HPP

#include "MPI_Node.hpp"

class ServerNode : public MPI_Node {
public:
    ServerNode();
    ServerNode(const ServerNode& orig);
    virtual ~ServerNode();
    
public:
    void loadVisualizationLib();
    
private:

};

#endif /* SERVERNODE_HPP */

