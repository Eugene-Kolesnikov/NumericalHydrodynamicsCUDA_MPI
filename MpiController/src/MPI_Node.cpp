/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   MPI_Node.cpp
 * Author: eugene
 *
 * Created on November 1, 2017, 1:13 PM
 */

#include "MPI_Node.hpp"
#include <dlfcn.h>
#include <list>
#include <map>
#include "../../ComputationalModel/src/computationalModel_interface.h"
#include "../../ComputationalModel/src/ComputationalModel.hpp"
#include <cmath> // floor


MPI_Node::MPI_Node(size_t globalRank, size_t totalNodes, std::string app_path):
    Log(globalRank)
{
    totalMPINodes = totalNodes;
    globalMPI_id = globalRank;
    appPath = app_path;

    MPI_NODES_X = 0;
    MPI_NODES_Y = 0;
    CUDA_Y_THREADS = 0;
    CUDA_Y_THREADS = 0;
    TAU = 0.0;
    TOTAL_TIME = 0.0;
    STEP_LENGTH = 0.0;
    N_X = 0;
    N_Y = 0;

    model = nullptr;

    parserLibHandle = nullptr;
    createConfig = nullptr;
    readConfig = nullptr;
    compModelLibHandle = nullptr;
    createComputationalModel = nullptr;
}

MPI_Node::~MPI_Node()
{
    if(parserLibHandle != nullptr)
        dlclose(parserLibHandle);
    if(compModelLibHandle != nullptr)
        dlclose(compModelLibHandle);
    if(model != nullptr)
        delete model;
}

void MPI_Node::initEnvironment()
{
    Log.openLogFile(appPath);
    loadXMLParserLib();
    loadComputationalModelLib();
    parseConfigFile();
    model->createMpiStructType();
}

void MPI_Node::loadXMLParserLib()
{
    /// Create a path to the lib
    std::string libpath = appPath + "libConfigParser.1.0.0.dylib";
    parserLibHandle = dlopen(libpath.c_str(), RTLD_LOCAL | RTLD_LAZY);
    if (parserLibHandle == nullptr) {
        throw std::runtime_error(dlerror());
    } else {
        Log << "Opened the parser dynamic library";
    }
    createConfig = (void (*)(void*, const char*))dlsym(parserLibHandle, "createConfig");
    readConfig = (void* (*)(const char*))dlsym(parserLibHandle, "readConfig");
    if(readConfig == nullptr || createConfig == nullptr) {
        throw std::runtime_error("Can't load functions from the XML parser library!");
    }
}

void MPI_Node::parseConfigFile()
{
    std::string filepath = appPath + "CONFIG";
    void* lst = readConfig(filepath.c_str());
    if(lst == nullptr)
        throw std::runtime_error("Configuration file is empty!");
    using namespace std;
    list<pair<string,double>>* params = (list<pair<string,double>>*)lst;
    std::string compModel;
    std::string gridModel;
    for(auto it = params->begin(); it != params->end(); ++it)
    {
        if(it->first == "MPI_NODES_X") {
            MPI_NODES_X = static_cast<size_t>(it->second);
        } else if(it->first == "MPI_NODES_Y") {
            MPI_NODES_Y = static_cast<size_t>(it->second);
        } else if(it->first == "CUDA_X_THREADS") {
            CUDA_X_THREADS = static_cast<size_t>(it->second);
        } else if(it->first == "CUDA_Y_THREADS") {
            CUDA_Y_THREADS = static_cast<size_t>(it->second);
        } else if(it->first == "TAU") {
            TAU = it->second;
        } else if(it->first == "TOTAL_TIME") {
            TOTAL_TIME = it->second;
        } else if(it->first == "STEP_LENGTH") {
            STEP_LENGTH = it->second;
        } else if(it->first == "N_X") {
            N_X = static_cast<size_t>(it->second);
        } else if(it->first == "N_Y") {
            N_Y = static_cast<size_t>(it->second);
        } else if(it->first == "LBM" && it->second == 1) {
            compModel = "LBM";
        } else if(it->first == "NS" && it->second == 1) {
            compModel = "NS";
        } else if(it->first == "USG" && it->second == 1) {
            gridModel = "USG";
        } else if(it->first == "STAG" && it->second == 1) {
            gridModel = "STAG";
        } else if(it->first == "RND_TR" && it->second == 1) {
            gridModel = "RND_TR";
        } else {
            // nothing
        }
    }
    if(compModel.empty() || gridModel.empty()) {
        throw std::runtime_error("Computational model and/or grid model are not"
                "specified in the configuration file!");
    }

    model = (ComputationalModel*)createComputationalModel(compModel.c_str(), gridModel.c_str());

    if(checkParsedParameters() == false) {
        throw std::runtime_error("Not all required parameters are read from the configuration file!");
    }

    lN_X = N_X / MPI_NODES_X;
    lN_Y = N_Y / MPI_NODES_Y;
    setLocalMPI_ids(globalMPI_id, localMPI_id_x, localMPI_id_y);
}

bool MPI_Node::checkParsedParameters()
{
    if(MPI_NODES_X == 0 || MPI_NODES_Y == 0 || CUDA_Y_THREADS == 0 ||
            CUDA_Y_THREADS == 0 || TAU == 0.0 || TOTAL_TIME == 0.0 ||
        STEP_LENGTH == 0.0 || N_X == 0 || N_Y == 0 || model == nullptr)
        return false;
    else
        return true;
}

void MPI_Node::loadComputationalModelLib()
{
    std::string libpath = appPath + "libComputationalModel.1.0.0.dylib";
    compModelLibHandle = dlopen(libpath.c_str(), RTLD_LOCAL | RTLD_LAZY);
    if (compModelLibHandle == nullptr) {
        throw std::runtime_error(dlerror());
    } else {
        Log << "Opened the computational model dynamic library";
    }
    createComputationalModel =
            (void* (*)(const char*, const char*))dlsym(compModelLibHandle, "createComputationalModel");
    if(createComputationalModel == nullptr) {
        throw std::runtime_error("Can't load the function from the Computational model library!");
    }
}

void MPI_Node::setComputationalModelEnv(ComputationalModel::NODE_TYPE node_type)
{
    model->setAppPath(appPath);
    model->setMPI_NODES_X(MPI_NODES_X);
    model->setMPI_NODES_Y(MPI_NODES_Y);
    model->setCUDA_X_THREADS(CUDA_X_THREADS);
    model->setCUDA_Y_THREADS(CUDA_Y_THREADS);
    model->setTAU(TAU);
    model->setNodeType(node_type);
    model->setN_X(N_X);
    model->setN_Y(N_Y);
    model->setLN_X(lN_X);
    model->setLN_Y(lN_Y);
    model->initializeField();
}

void MPI_Node::setLocalMPI_ids(const size_t globalId, size_t& localIdx, size_t& localIdy)
{
    localIdy = static_cast<size_t>(floor(globalId / MPI_NODES_X));
    localIdx = globalId - localIdy * MPI_NODES_X;
}

void MPI_Node::setLocalMPI_ids(const int globalId, int& localIdx, int& localIdy)
{
    if(globalId < 0) {
        localIdx = MPI_NODES_X;
        localIdy = MPI_NODES_Y;
    } else {
        localIdy = static_cast<size_t>(floor(globalId / MPI_NODES_X));
        localIdx = globalId - localIdy * MPI_NODES_X;
    }
}

size_t MPI_Node::getGlobalMPIid(size_t mpi_id_x, size_t mpi_id_y)
{
    return mpi_id_y * MPI_NODES_X + mpi_id_x;
}
