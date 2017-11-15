#include "TestScheme.hpp"

#include <string>
#include <exception>
#include <cmath>

__global__ void performGPUSimulationStep_kernel(Cell* cu_field, Cell* cu_lr_halo,
		Cell* cu_tb_halo, Cell* cu_lrtb_halo, size_t N_X, size_t N_Y)
{
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
	double k = cu_field[tid_y * N_X + tid_x].r;
	cu_field[tid_y * N_X + tid_x].r = k + 0.01;
	__syncthreads();
}

__global__ void updateGPUGlobalBorders_kernel(Cell* cu_field, Cell* cu_lr_halo,
		Cell* cu_tb_halo, Cell* cu_lrtb_halo, size_t N_X, size_t N_Y, size_t type)
{

}

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
    HANDLE_CUERROR_PTR( cudaHostAlloc((void**)&ptr, N_X * N_Y * sizeof(Cell), cudaHostAllocDefault) );
    return (void*)ptr;
}

void* TestScheme::createGPUField(size_t N_X, size_t N_Y)
{
    Cell* ptr;
    HANDLE_CUERROR_PTR( cudaMalloc((void**)&ptr, N_X * N_Y * sizeof(Cell)) );
    return (void*)ptr;
}

ErrorStatus TestScheme::initField(void* field, size_t N_X, size_t N_Y)
{
    Cell* cfield = (Cell*)field;
    size_t global;
    for(size_t x = 0; x < N_X; ++x) {
        for(size_t y = 0; y < N_Y; ++y) {
            global = y * N_X + x;
            cfield[global].r = ((double)x+(double)y)/((double)N_X+(double)N_Y)/2;
            cfield[global].u = 0.0;
            cfield[global].v = 0.0;
            cfield[global].e = 0.0;
        }
    }
	return GPU_SUCCESS;
}

void* TestScheme::initHalos(size_t N)
{
    return (void*)(new Cell[N]);
}

void* TestScheme::initPageLockedHalos(size_t N)
{
    Cell* ptr;
    HANDLE_CUERROR_PTR( cudaHostAlloc((void**)&ptr, N * sizeof(Cell), cudaHostAllocDefault) );
    return (void*)ptr;
}

void* TestScheme::initGPUHalos(size_t N)
{
    Cell* ptr;
    HANDLE_CUERROR_PTR( cudaMalloc((void**)&ptr, N * sizeof(Cell)) );
    return (void*)ptr;
}

ErrorStatus TestScheme::performCPUSimulationStep(void* tmpCPUField, void* lr_halo,
        void* tb_halo, void* lrtb_halo, size_t N_X, size_t N_Y)
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
	return GPU_SUCCESS;
}

ErrorStatus TestScheme::performGPUSimulationStep(void* cu_field, void* cu_lr_halo,
        void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
        size_t CUDA_X_BLOCKS, size_t CUDA_Y_BLOCKS, size_t CUDA_X_THREADS,
        size_t CUDA_Y_THREADS, void* stream)
{
	size_t SharedMemoryPerBlock = CUDA_X_THREADS * CUDA_Y_THREADS * sizeof(Cell);
	float blocksPerSM = ceil((float)CUDA_X_BLOCKS * (float)CUDA_Y_BLOCKS / (float)amountSMs);
	size_t totalSharedMemoryPerBlock = ceil((float)totalSharedMemoryPerSM / blocksPerSM);
	/// Check if there is enough shared memory
	if(totalSharedMemoryPerBlock < SharedMemoryPerBlock) {
		errorString = std::string("Trying to allocate too much CUDA shared memory: ") +
			std::to_string(totalSharedMemoryPerBlock) + std::string(" bytes is available per block, ") +
			std::to_string(SharedMemoryPerBlock) + std::string(" bytes per block is requested!");
		return GPU_ERROR;
	}
    cudaStream_t* cuStream = (cudaStream_t*)stream;
	performGPUSimulationStep_kernel <<< dim3(CUDA_X_BLOCKS, CUDA_Y_BLOCKS, 1),
		dim3(CUDA_X_THREADS, CUDA_Y_THREADS, 1), SharedMemoryPerBlock,
		*cuStream >>> ((Cell*)cu_field, (Cell*)cu_lr_halo, (Cell*)cu_tb_halo,
            (Cell*)cu_lrtb_halo, N_X, N_Y);
	/// Check if the kernel executed without errors
	lastCudaError = cudaGetLastError();
	if(lastCudaError != cudaSuccess) {
		errorString = std::string("performGPUSimulationStep: ") +
			std::string(cudaGetErrorString(lastCudaError));
		return GPU_ERROR;
	}
	return GPU_SUCCESS;
}

ErrorStatus TestScheme::updateGPUGlobalBorders(void* cu_field, void* cu_lr_halo,
            void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
            size_t type, size_t CUDA_X_BLOCKS, size_t CUDA_Y_BLOCKS,
            size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS, void* stream)
{
    cudaStream_t* cuStream = (cudaStream_t*)stream;
	updateGPUGlobalBorders_kernel <<< dim3(CUDA_X_BLOCKS, CUDA_Y_BLOCKS, 1),
		dim3(CUDA_X_THREADS, CUDA_Y_THREADS, 1), 0,
		*cuStream >>> ((Cell*)cu_field, (Cell*)cu_lr_halo, (Cell*)cu_tb_halo,
            (Cell*)cu_lrtb_halo, N_X, N_Y, type);
	/// Check if the kernel executed without errors
	lastCudaError = cudaGetLastError();
	if(lastCudaError != cudaSuccess) {
		errorString = std::string("performGPUSimulationStep: ") +
			std::string(cudaGetErrorString(lastCudaError));
		return GPU_SUCCESS;
	}
	return GPU_SUCCESS;
}

void* TestScheme::getMarkerValue()
{
    return (void*)(&marker);
}
