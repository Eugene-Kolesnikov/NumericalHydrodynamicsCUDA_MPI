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
#include "../../ComputationalModel/src/ComputationalModel.hpp" // ComputationalModel::NODE_TYPE
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
        DLV_init(N_X, N_Y, OUTPUT_OPTION::MPEG, (appPath + "../../../").c_str());
    } catch(const std::runtime_error& err) {
        if(Log.is_open())
            Log << _ERROR_ << std::string("(ServerNode:initEnvironment): ") + err.what();
        throw;
    }
}

void ServerNode::runNode()
{
    try {
        // Continue loading updated fields and visualize them
        // until the stop marker is set
        while(model->checkStopMarker() == false) {
            DLV_visualize(model->getField(), N_X, N_Y);
            loadUpdatedSubfields();
        }
        // Finish the visualization process
        if(!DLV_terminate())
            throw std::runtime_error("Visualization library was not able"
                    " to terminate successfully!");
        MPI_Node::finalBarrierSync();
    } catch(const std::runtime_error& err) {
        Log << _ERROR_ << std::string("(ServerNode:runNode): ") + err.what();
        throw;
    }
}

void ServerNode::loadVisualizationLib()
{
    std::string libpath = appPath + "libVisuzalization.1.0.0.dylib";
    void* m_visualizationLibHandle = dlopen(libpath.c_str(), RTLD_LOCAL | RTLD_LAZY);
    if (m_visualizationLibHandle == nullptr) {
        throw std::runtime_error(dlerror());
    } else {
        Log << "Opened the dynamic visualization library";
    }

    DLV_init = (bool (*)(size_t, size_t, 
            enum OUTPUT_OPTION, const char*))dlsym(m_visualizationLibHandle, "DLV_init");
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
            // Since the MPI is working with the global 1D array of MPI indexes,
            // the 2D MPI id must be converted to the global MPI
            globalMPIidReceiver = getGlobalMPIid(mpi_node_x, mpi_node_y);
            Log << "Trying to send " + std::to_string(totalAmountCellsToTransfer) +
                " amount of field cells to the node (" + 
                std::to_string(mpi_node_x) + "," + 
                std::to_string(mpi_node_y) + "), (global id: " +
                std::to_string(globalMPIidReceiver) + ")";
            mpi_err_status = MPI_Send(tmpStoragePtr, totalAmountCellsToTransfer, 
                model->MPI_CellType, globalMPIidReceiver, globalMPIidReceiver, MPI_COMM_WORLD);
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
    // make sure that every ComputationalNode received its part of data
    MPI_Barrier(MPI_COMM_WORLD); 
    Log << "Barrier synchronization has been successfully performed.";
}

void ServerNode::loadUpdatedSubfields()
{
    MPI_Status status;
    size_t totalAmountCellsToTransfer = lN_X * lN_Y;
    // The temporary storage doesn't change, so the pointer must be
    // obtained only once.
    void* tmpStoragePtr = model->getTmpCPUFieldStoragePtr();
    for(size_t mpi_node_x = 0; mpi_node_x < MPI_NODES_X; ++mpi_node_x) {
        for(size_t mpi_node_y = 0; mpi_node_y < MPI_NODES_Y; ++mpi_node_y) {
            size_t globalMPIidSender = getGlobalMPIid(mpi_node_x, mpi_node_y);
            Log << "Trying to receive " + std::to_string(totalAmountCellsToTransfer) +
                " amount of field cells from the node (" + 
                std::to_string(mpi_node_x) + "," + 
                std::to_string(mpi_node_y) + "), (global id: " +
                std::to_string(globalMPIidSender) + ")";
            MPI_Recv(tmpStoragePtr, totalAmountCellsToTransfer, model->MPI_CellType, 
                    globalMPIidSender, globalMPIidSender, MPI_COMM_WORLD, &status);
            // After receiving the message, check the status to determine
            // how many numbers were actually received
            int number_amount;
            MPI_Get_count(&status, model->MPI_CellType, &number_amount);
            if(number_amount != totalAmountCellsToTransfer) {
                Log << _WARNING_ << ("Received " + std::to_string(number_amount) + 
                        " amount of field cells instead of " + std::to_string(totalAmountCellsToTransfer));
            }
            Log << "Data has been successfully received from the node (" + 
                std::to_string(mpi_node_x) + "," + 
                std::to_string(mpi_node_y) + "), (global id: " +
                std::to_string(globalMPIidSender) + ")";
            // On the next iteration the values of the array to which the 
            // tmpStoragePtr is referenced will be changed, so it is 
            // important to update the global field first.
            model->updateGlobalField(mpi_node_x, mpi_node_y);
        }
    }
    Log << "Data has been successfully received from all computational nodes";
    // make sure that every ComputationalNode sent their subfields
    MPI_Barrier(MPI_COMM_WORLD);
    Log << "Barrier synchronization has been successfully performed.";
}
