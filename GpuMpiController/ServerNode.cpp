/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   ServerNode.cpp
 * Author: eugene
 * 
 * Created on November 1, 2017, 12:49 PM
 */

#include "ServerNode.hpp"
#include "ComputationalModel.hpp" // ComputationalModel::NODE_TYPE
#include <dlfcn.h>
#include <mpi.h>

ServerNode::ServerNode(size_t globalRank, size_t totalNodes, std::string app_path):
    MPI_Node(globalRank, totalNodes, app_path)
{
    m_visualizationLibHandle = nullptr;
    DLV_init = nullptr;
    DLV_visualize = nullptr;
    DLV_terminate = nullptr;
}

ServerNode::~ServerNode() 
{
    if(m_visualizationLibHandle != nullptr)
        dlclose(m_visualizationLibHandle); 
}

void ServerNode::initEnvironment()
{
    try {
        MPI_Node::initEnvironment();
        loadVisualizationLib();
        MPI_Node::setComputationalModelEnv(ComputationalModel::NODE_TYPE::SERVER_NODE);
        sendInitFieldToCompNodes();
    } catch(std::runtime_error err) {
        if(Log.is_open())
            Log << _ERROR_ << "(ServerNode:initEnvironment): " << err.what();
        throw err;
    }
}

void ServerNode::runNode()
{
    try {
        if(DLV_init(N_X, N_Y, OUTPUT_OPTION::MPEG == false))
            throw std::runtime_error("Visualization library was not able"
                    " to initialize successfully!");
        
        // \TODO: Place for the code of the server node
        
        if(!DLV_terminate())
            throw std::runtime_error("Visualization library was not able"
                    " to terminate successfully!");
    } catch(std::runtime_error err) {
        Log << _ERROR_ << "(ServerNode:runNode): " << err.what();
        throw err;
    }
}

void ServerNode::loadVisualizationLib()
{
    void* m_visualizationLibHandle = dlopen("libVisuzalization.1.0.0.dylib", RTLD_LOCAL | RTLD_LAZY);
    if (m_visualizationLibHandle == nullptr) {
        throw std::runtime_error(dlerror());
    } else {
        Log << "Opened the dynamic visualization library";
    }

    DLV_init = (bool (*)(size_t, size_t, 
            enum OUTPUT_OPTION))dlsym(m_visualizationLibHandle, "DLV_init");
    DLV_visualize = (bool (*)(void*,
        size_t, size_t))dlsym(m_visualizationLibHandle, "DLV_visualize");
    DLV_terminate = (bool (*)())dlsym(m_visualizationLibHandle, "DLV_terminate");

    if(DLV_init == nullptr || DLV_visualize == nullptr || DLV_terminate == nullptr) {
        throw std::runtime_error("Can't load functions from the visualization library!");
    }
}

void ServerNode::sendInitFieldToCompNodes()
{
    Log << "Sharing the initialized field";
    size_t totalAmountCellsToTransfer = lN_X * lN_Y;
    void* tmpStoragePtr = nullptr;
    size_t globalMPIidReceiver = 0;
    int mpi_err_status, resultlen;
    char err_buffer[MPI_MAX_ERROR_STRING];
    // sending data to computational nodes
    for(size_t mpi_node_x = 0; mpi_node_x < MPI_NODES_X; ++mpi_node_x) {
        for(size_t mpi_node_y = 0; mpi_node_y < MPI_NODES_Y; ++mpi_node_y) {
            Log << "Sending a request to prepare a subfield for node (" + 
                std::to_string(mpi_node_x) + "," + 
                std::to_string(mpi_node_y) + ")";
            model->prepareSubfield(mpi_node_x, mpi_node_y);
            Log << "Preparation of a subfield successfully performed";
            Log << "Sending a request for a temporary CPU field storage"
                    " pointer with the prepared subfield";
            tmpStoragePtr = model->getTmpCPUFieldStoragePtr();
            Log << "Received the temporary CPU field storage pointer";
            // since the MPI is working with global 1D array of MPI indexes,
            // the 2D MPI id must be converted to the global MPI
            globalMPIidReceiver = getGlobalMPIid(mpi_node_x, mpi_node_y);
            Log << "Trying to send " + std::to_string(totalAmountCellsToTransfer) +
                " amount of field cells to node (" + 
                std::to_string(mpi_node_x) + "," + 
                std::to_string(mpi_node_y) + "), (global id: " +
                std::to_string(globalMPIidReceiver) + ")";
            mpi_err_status = MPI_Send(tmpStoragePtr, totalAmountCellsToTransfer, 
                    model->MPI_CellType, globalMPIidReceiver, 0, MPI_COMM_WORLD);
            // Check if the MPI transfer was successful
            if(mpi_err_status != MPI_SUCCESS) {
                MPI_Error_string(mpi_err_status, err_buffer, &resultlen);
                throw std::runtime_error(err_buffer);
            }
            Log << "Data has been successfully sent to the node (" + 
                std::to_string(mpi_node_x) + "," + 
                std::to_string(mpi_node_y) + "), (global id: " +
                std::to_string(globalMPIidReceiver) + ")";
        }
    }
    Log << "Subfields has been successfully sent to all computational nodes";
    MPI_Barrier(MPI_COMM_WORLD); // make sure that every one has it's part of data
    Log << "Barrier synchronization has been successfully performed.";
}
