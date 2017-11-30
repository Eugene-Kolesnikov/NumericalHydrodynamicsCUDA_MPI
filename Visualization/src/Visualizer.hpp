#ifndef VISUALIZATION_HPP
#define VISUALIZATION_HPP

#include <cstdlib>
#include <string>
#include <list>
#include <map>
#include <vector>
#include "../../MpiController/src/FileLogger.hpp"

using namespace std;

class Visualizer
{
public:
    Visualizer(logging::FileLogger* _Log);
    virtual ~Visualizer();

public:
    void setEnvironment(const string& _appPath, size_t _MPI_NODES_X,
      size_t _MPI_NODES_Y, size_t _CUDA_X_THREADS, size_t _CUDA_Y_THREADS,
      double _TAU, double _TOTAL_TIME, double _STEP_LENGTH, size_t _N_X, size_t _N_Y,
      size_t _X_MAX, size_t _Y_MAX, const list<pair<string,size_t>>& _params,
      size_t _size_of_datatype, size_t _nitems);

public:
    virtual void initVisualizer() = 0;
    virtual void renderFrame(void* field) = 0;
    virtual void setProgress(double val) = 0;
    virtual void deinitVisualizer() = 0;

protected:
    string appPath;
    size_t MPI_NODES_X;
    size_t MPI_NODES_Y;
    size_t CUDA_X_THREADS;
    size_t CUDA_Y_THREADS;
    double TAU;
    double TOTAL_TIME;
    double STEP_LENGTH;
    size_t N_X;
    size_t N_Y;
    size_t X_MAX;
    size_t Y_MAX;

protected:
    size_t size_of_datatype;
    size_t nitems;
    list<pair<string,size_t>> params;
    vector<size_t> ids;

protected:
    logging::FileLogger* Log;
};

#endif // VISUALIZATION_HPP
