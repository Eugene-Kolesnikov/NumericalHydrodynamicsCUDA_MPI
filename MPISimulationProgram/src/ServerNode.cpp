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

#include <MPISimulationProgram/include/ServerNode.hpp>
#include <ComputationalModel/include/ComputationalModel.hpp> // ComputationalModel::NODE_TYPE
#include <dlfcn.h>
#include <mpi.h>

ServerNode::ServerNode(size_t globalRank, size_t totalNodes, std::string app_path, int* _argc, char** _argv):
    MPI_Node(globalRank, totalNodes, app_path, _argc, _argv)
{
    visLibHandler = nullptr;
    visualizer = nullptr;
}

ServerNode::~ServerNode()
{
    if(visualizer != nullptr)
        delete visualizer;
    libLoader::close(visLibHandler);
}

void ServerNode::initEnvironment()
{
    try {
        MPI_Node::initEnvironment();
        loadVisualizationLib();
        MPI_Node::setComputationalModelEnv(ComputationalModel::NODE_TYPE::SERVER_NODE);
        model->initializeField();
        sendInitFieldToCompNodes();
        visualizer->setEnvironment(appPath, MPI_NODES_X, MPI_NODES_Y,
            CUDA_X_THREADS, CUDA_Y_THREADS, TAU, TOTAL_TIME, STEP_LENGTH,
            N_X, N_Y, X_MAX, Y_MAX, model->getScheme()->getDrawParams(),
            model->getScheme()->getSizeOfDatastruct(), model->getScheme()->getNumberOfElements());
        Log << "Set the visualizer's environment";
        visualizer->initVisualizer();
        Log << "Initialized the visualizer";
        visualizer->renderFrame(model->getField());
        Log << "Rendered the initial frame";
    } catch(const std::runtime_error& err) {
        if(Log.is_open())
            Log << _ERROR_ << std::string("(ServerNode:initEnvironment): ") + err.what();
        throw;
    }
}

void ServerNode::runNode()
{
    try {
        size_t counter = 0;
        double amountOfSteps = TOTAL_TIME / TAU;
        Log << "Starting the rendering loop";
        // Continue loading updated fields and visualize them
        // until the stop marker is set
        while(model->checkStopMarker() == false) {
            loadUpdatedSubfields();
            visualizer->setProgress(counter * (double)STEP_LENGTH / amountOfSteps * 100.0);
            Log << "Set the progress";
            visualizer->renderFrame(model->getField());
            Log << "Rendered the frame";
            counter++;
        }
        // Finish the visualization process
        visualizer->deinitVisualizer();
        Log << "Visualizer has been deinitialized";
        MPI_Node::finalBarrierSync();
        model->deinitModel();
        Log << "Computational model has been deinitialized";
    } catch(const std::runtime_error& err) {
        Log << _ERROR_ << std::string("(ServerNode:runNode): ") + err.what();
        throw;
    }
}

void ServerNode::loadVisualizationLib()
{
    std::string libpath = appPath + SystemRegister::VisLib::name;
    visLibHandler = libLoader::open(libpath);
    auto _createVisualizer = libLoader::resolve<decltype(&createVisualizer)>(visLibHandler, SystemRegister::VisLib::interface);
    Log << "Opened the dynamic visualization library";
    visualizer = reinterpret_cast<Visualizer*>(_createVisualizer(argc, argv, &Log));
    Log << "Created a visualizer";
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
    mpi_err_status = MPI_Barrier(MPI_COMM_WORLD);
    // Check if the MPI barrier synchronization was successful
    if(mpi_err_status != MPI_SUCCESS) {
        MPI_Error_string(mpi_err_status, err_buffer, &resultlen);
        throw std::runtime_error(err_buffer);
    }
    Log << "Barrier synchronization has been successfully performed.";
}

void ServerNode::loadUpdatedSubfields()
{
    MPI_Status status;
    int mpi_err_status, resultlen;
    char err_buffer[MPI_MAX_ERROR_STRING];
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
    mpi_err_status = MPI_Barrier(MPI_COMM_WORLD);
    // Check if the MPI barrier synchronization was successful
    if(mpi_err_status != MPI_SUCCESS) {
        MPI_Error_string(mpi_err_status, err_buffer, &resultlen);
        throw std::runtime_error(err_buffer);
    }
    Log << "Barrier synchronization has been successfully performed.";
}
