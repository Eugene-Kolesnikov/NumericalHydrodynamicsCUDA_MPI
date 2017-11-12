#include "TestScheme.hpp"

TestScheme::TestScheme():
    ComputationalScheme()
{
    marker = -1.0;
}

const std::type_info& TestScheme::getDataTypeid()
{
    return typeid(STRUCT_DATA_TYPE);
}

size_t TestScheme::getSizeOfDatatype()
{
    return sizeof(STRUCT_DATA_TYPE);
}

size_t TestScheme::getNumberOfElements()
{
    return static_cast<size_t>(sizeof(Cell) / sizeof(STRUCT_DATA_TYPE));
}

void* TestScheme::createField(size_t N_X, size_t N_Y)
{
    return (void*)(new Cell[N_X * N_Y]);
}

void* TestScheme::createPageLockedField(size_t N_X, size_t N_Y)
{
    Cell* ptr;
    HANDLE_CUERROR( cudaHostAlloc((void**)&ptr, N_X * N_Y * sizeof(Cell), cudaHostAllocDefault) );
    return (void*)ptr;
}

void* TestScheme::createGPUField(size_t N_X, size_t N_Y)
{
    Cell* ptr;
    HANDLE_CUERROR( cudaMalloc((void**)&ptr, N_X * N_Y * sizeof(Cell)) );
    return (void*)ptr;
}

void TestScheme::initField(void* field, size_t N_X, size_t N_Y)
{
    Cell* cfield = (Cell*)field;
    size_t global;
    for(size_t x = 0; x < N_X; ++x) {
        for(size_t y = 0; y < N_Y; ++y) {
            global = y * N_X + x;
            cfield[global].r = drand48();
            cfield[global].u = 0.0;
            cfield[global].v = 0.0;
            cfield[global].e = 0.0;
        }
    }
}

void* TestScheme::initHalos(size_t N)
{
    return (void*)(new Cell[N]);
}

void* TestScheme::initPageLockedHalos(size_t N)
{
    Cell* ptr;
    HANDLE_CUERROR( cudaHostAlloc((void**)&ptr, N * sizeof(Cell), cudaHostAllocDefault) );
    return (void*)ptr;
}

void* TestScheme::initGPUHalos(size_t N)
{
    Cell* ptr;
    HANDLE_CUERROR( cudaMalloc((void**)&ptr, N * sizeof(Cell)) );
    return (void*)ptr;
}

void TestScheme::performCPUSimulationStep(void* tmpCPUField, void* lr_halo, 
        void* tb_halo, size_t N_X, size_t N_Y)
{
    Cell* ctmpCPUField = (Cell*)tmpCPUField;
    size_t global, global1;
    Cell* tmpField = new Cell[N_Y];
    for(size_t y = 0; y < N_Y; ++y) {
        global = y * N_X;
        tmpField[y].r = ctmpCPUField[global].r;
        tmpField[y].u = ctmpCPUField[global].u;
        tmpField[y].v = ctmpCPUField[global].v;
        tmpField[y].e = ctmpCPUField[global].e;
    }
    for(size_t x = 1; x < N_X; ++x) {
        for(size_t y = 0; y < N_Y; ++y) {
            global = y * N_X + x;
            global1 = y * N_X + x - 1;
            ctmpCPUField[global1].r = ctmpCPUField[global].r;
            ctmpCPUField[global1].u = ctmpCPUField[global].u;
            ctmpCPUField[global1].v = ctmpCPUField[global].v;
            ctmpCPUField[global1].e = ctmpCPUField[global].e;
        }
    }
    for(size_t y = 1; y <= N_Y; ++y) {
        global = y * N_X - 1;
        ctmpCPUField[global].r = tmpField[y-1].r;
        ctmpCPUField[global].u = tmpField[y-1].u;
        ctmpCPUField[global].v = tmpField[y-1].v;
        ctmpCPUField[global].e = tmpField[y-1].e;
    }
    delete[] tmpField;
}

void* TestScheme::getMarkerValue()
{
    return (void*)(&marker);
}
