#include "Visualizer.hpp"

Visualizer::Visualizer(logging::FileLogger* _Log): Log(_Log)
{
}

Visualizer::~Visualizer()
{
}

void Visualizer::setEnvironment(const string& _appPath, size_t _MPI_NODES_X,
    size_t _MPI_NODES_Y, size_t _CUDA_X_THREADS, size_t _CUDA_Y_THREADS,
    double _TAU, double _TOTAL_TIME, double _STEP_LENGTH, size_t _N_X, size_t _N_Y,
    size_t _X_MAX, size_t _Y_MAX, const list<pair<string,size_t>>& _params,
    size_t _size_of_datatype, size_t _nitems)
{
    /// Initialize the simulation environment
    appPath = _appPath;
    MPI_NODES_X = _MPI_NODES_X;
    MPI_NODES_Y = _MPI_NODES_Y;
    CUDA_X_THREADS = _CUDA_X_THREADS;
    CUDA_Y_THREADS = _CUDA_Y_THREADS;
    TAU = _TAU;
    TOTAL_TIME = _TOTAL_TIME;
    STEP_LENGTH = _STEP_LENGTH;
    N_X = _N_X;
    N_Y = _N_Y;
    X_MAX = _X_MAX;
    Y_MAX = _Y_MAX,
    /// Initialize the scheme specific variables
    params = _params;
    size_of_datatype = _size_of_datatype;
    nitems = _nitems;
    /// Save field ids which must be saved to the report file
    for(auto it = params.begin(); it != params.end(); ++it) {
        ids.push_back(it->second);
    }
}
