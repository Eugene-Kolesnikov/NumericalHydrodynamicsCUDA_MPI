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

#include <MPISimulationProgram/include/MPI_Node.hpp>
#include <dlfcn.h>
#include <list>
#include <map>
#include <ComputationalModel/include/interface.h>
#include <ComputationalModel/include/ComputationalModel.hpp>
#include <ConfigParser/include/interface.h>
#include <cmath> // floor


MPI_Node::MPI_Node(size_t globalRank, size_t totalNodes, std::string app_path, int* _argc, char** _argv):
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
    X_MAX = 0;
    Y_MAX = 0;

    model = nullptr;

    parserLibHandler = nullptr;
    compModelLibHandler = nullptr;

    argc = _argc;
    argv = _argv;
}

MPI_Node::~MPI_Node()
{
    libLoader::close(parserLibHandler);
    if(model != nullptr)
        delete model;
    libLoader::close(compModelLibHandler);
}

void MPI_Node::initEnvironment()
{
    Log.openLogFile(appPath);
    loadXMLParserLib();
    parseSystemRegister();
    loadComputationalModelLib();
    parseConfigFile();
    model->createMpiStructType();
}

void MPI_Node::loadXMLParserLib()
{
    /// Create a path to the lib
    std::string libpath = appPath + SystemRegister::ConfigParser::name;
    parserLibHandler = libLoader::open(libpath);
    Log << "Opened the config parser dynamic library";
}

/// TODO: Change the function to parsing a system configuration file
void MPI_Node::parseSystemRegister()
{
    SystemRegister::ConfigFile = "CONFIG.xml";
    SystemRegister::VisLib::name = "libVisualization.2.0.0.dylib";
    SystemRegister::VisLib::interface = "createVisualizer";
    SystemRegister::CompModel::name = "libComputationalModel.1.0.0.so";
    SystemRegister::CompModel::interface = "createComputationalModel";
    SystemRegister::CompScheme::name = "libComputationalScheme.1.0.0.so";
    SystemRegister::CompScheme::interface = "createScheme";
}

void MPI_Node::parseConfigFile()
{
    auto _readConfig = libLoader::resolve<decltype(&readConfig)>(parserLibHandler,
        SystemRegister::ConfigParser::interface[SystemRegister::ConfigParser::interfaceFunctions::readConfig]);
    std::string filepath = appPath + SystemRegister::ConfigFile;
    void* lst = _readConfig(filepath.c_str());
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
        } else if(it->first == "X_MAX") {
            X_MAX = it->second;
        } else if(it->first == "Y_MAX") {
            Y_MAX = it->second;
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

    auto _createComputationalModel = libLoader::resolve<decltype(&createComputationalModel)>(compModelLibHandler, SystemRegister::CompModel::interface);
    model = reinterpret_cast<ComputationalModel*>(_createComputationalModel(compModel.c_str(), gridModel.c_str()));
    Log << "Computational model has been successfully created";

    if(checkParsedParameters() == false) {
        throw std::runtime_error("Not all required parameters are read from the configuration file!");
    }

    model->setAppPath(appPath);
    model->setLog(&Log);
    model->initScheme();

    lN_X = N_X / MPI_NODES_X;
    lN_Y = N_Y / MPI_NODES_Y;
    setLocalMPI_ids(globalMPI_id, localMPI_id_x, localMPI_id_y);
    Log << "Successfully parsed the configuration file";
}

bool MPI_Node::checkParsedParameters()
{
    if(MPI_NODES_X == 0 || MPI_NODES_Y == 0 || CUDA_Y_THREADS == 0 ||
            CUDA_Y_THREADS == 0 || TAU == 0.0 || TOTAL_TIME == 0.0 ||
            STEP_LENGTH == 0.0 || N_X == 0 || N_Y == 0 || X_MAX == 0 ||
            Y_MAX == 0 || model == nullptr)
        return false;
    else
        return true;
}

void MPI_Node::loadComputationalModelLib()
{
    std::string libpath = appPath + SystemRegister::CompModel::name;
    compModelLibHandler = libLoader::open(libpath);
    Log << "Opened the computational model dynamic library";
}

void MPI_Node::setComputationalModelEnv(ComputationalModel::NODE_TYPE node_type)
{
    model->setMPI_NODES_X(MPI_NODES_X);
    model->setMPI_NODES_Y(MPI_NODES_Y);
    model->setCUDA_X_THREADS(CUDA_X_THREADS);
    model->setCUDA_Y_THREADS(CUDA_Y_THREADS);
    model->setTAU(TAU);
    model->setNodeType(node_type);
    model->setN_X(N_X);
    model->setN_Y(N_Y);
    model->setX_MAX(X_MAX);
    model->setY_MAX(Y_MAX);
    model->setLN_X(lN_X);
    model->setLN_Y(lN_Y);
    model->initializeEnvironment();
}

void MPI_Node::setLocalMPI_ids(const size_t globalId, size_t& localIdx, size_t& localIdy)
{
    localIdy = static_cast<size_t>(floor(globalId / MPI_NODES_X));
    localIdx = globalId - localIdy * MPI_NODES_X;
}

void MPI_Node::setLocalMPI_ids(const int globalId, int& localIdx, int& localIdy)
{
    if(globalId == -1) {
        localIdx = -1;
        localIdy = -1;
    } else {
        localIdy = static_cast<size_t>(floor(globalId / MPI_NODES_X));
        localIdx = globalId - localIdy * MPI_NODES_X;
    }
}

int MPI_Node::getGlobalMPIid(int mpi_id_x, int mpi_id_y)
{
    if(mpi_id_x != -1 && mpi_id_y != -1)
        return mpi_id_y * MPI_NODES_X + mpi_id_x;
    else
        return -1;
}

void MPI_Node::finalBarrierSync()
{
    int mpi_err_status, resultlen;
    char err_buffer[MPI_MAX_ERROR_STRING];
    mpi_err_status = MPI_Barrier(MPI_COMM_WORLD);
    // Check if the MPI barrier synchronization was successful
    if(mpi_err_status != MPI_SUCCESS) {
        MPI_Error_string(mpi_err_status, err_buffer, &resultlen);
        throw std::runtime_error(err_buffer);
    }
    Log << "Simulation has been successfully finished";
}
