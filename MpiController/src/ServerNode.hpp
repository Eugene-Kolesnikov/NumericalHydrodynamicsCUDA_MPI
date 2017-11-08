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
#include "../../Visualization/src/interface.h"
#include <string>
#include <cstdlib>

class ServerNode : public MPI_Node {
public:
    ServerNode(size_t globalRank, size_t totalNodes, std::string app_path);
    virtual ~ServerNode();
    
    virtual void initEnvironment();
    virtual void runNode();
    
protected:
    void loadVisualizationLib();
    void sendInitFieldToCompNodes();
    void loadUpdatedSubfields();
    
protected:
    void* m_visualizationLibHandle;
     bool (*DLV_init)(size_t N_X, size_t N_Y, enum OUTPUT_OPTION outOption, const char* path);
     bool (*DLV_visualize)(void* field, size_t N_X, size_t N_Y);
     bool (*DLV_terminate)();
};

#endif /* SERVERNODE_HPP */

