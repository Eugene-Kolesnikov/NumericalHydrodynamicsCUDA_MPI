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

#include <string>
#include <cstdlib>
#include <utilities/Logger/include/FileLogger.hpp>
#include <ComputationalModel/include/GPU_Status.h>
#include <ComputationalModel/include/ComputationalModel.hpp>

class MPI_Node {
public:
    MPI_Node(size_t globalRank, size_t totalNodes, std::string app_path, int* _argc, char** _argv);
    virtual ~MPI_Node();

public:
    virtual void initEnvironment();
    virtual void runNode() = 0;

protected:
    void loadXMLParserLib();
    void parseConfigFile();
    bool checkParsedParameters();
    void loadComputationalModelLib();
    void setComputationalModelEnv(ComputationalModel::NODE_TYPE node_type);
    void setLocalMPI_ids(const size_t globalId, size_t& localIdx, size_t& localIdy);
    void setLocalMPI_ids(const int globalId, int& localIdx, int& localIdy);
    int getGlobalMPIid(int mpi_id_x, int mpi_id_y);
    void finalBarrierSync();

protected:
    void* parserLibHandle;
    void (*createConfig)(void* params, const char* filepath);
    void* (*readConfig)(const char* filepath);

protected:
    std::string appPath;
    logging::FileLogger Log;
    size_t totalMPINodes;
    size_t globalMPI_id;
    size_t localMPI_id_x;
    size_t localMPI_id_y;

protected:
    void* compModelLibHandle;
    void* (*createComputationalModel)(const char* compModel, const char* gridModel);
    ComputationalModel* model;

protected: // Configuration parameters
    size_t MPI_NODES_X;
    size_t MPI_NODES_Y;
    size_t CUDA_X_THREADS;
    size_t CUDA_Y_THREADS;
    double TAU;
    double TOTAL_TIME;
    double STEP_LENGTH;
    size_t N_X;
    size_t N_Y;
    double X_MAX;
    double Y_MAX;
    size_t lN_X; // local N_X for computational nodes
    size_t lN_Y; // local N_Y for computational nodes

protected:
    int* argc;
    char** argv;
};

#endif /* MPI_NODE_HPP */
