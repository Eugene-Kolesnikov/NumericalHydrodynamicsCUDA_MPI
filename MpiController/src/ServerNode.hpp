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
#include "../../Visualization/src/interface_visualization.h"
#include "../../Visualization/src/Visualizer.hpp"
#include <string>
#include <cstdlib>

class ServerNode : public MPI_Node {
public:
    ServerNode(size_t globalRank, size_t totalNodes, std::string app_path, int* _argc, char** _argv);
    virtual ~ServerNode();

    virtual void initEnvironment();
    virtual void runNode();

protected:
    void loadVisualizationLib();
    void sendInitFieldToCompNodes();
    void loadUpdatedSubfields();

protected:
    void* m_visualizationLibHandle;
    void* (*createVisualizer)(int* argc, char** argv, void* Log);
    Visualizer* visualizer;
};

#endif /* SERVERNODE_HPP */
