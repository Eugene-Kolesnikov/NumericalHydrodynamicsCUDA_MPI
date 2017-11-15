/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "computationalScheme_interface.h"
#include "TestScheme.hpp"

__global__ void create_GPUComputationalScheme(ComputationalScheme_GPU* scheme)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
		scheme = new TestScheme_GPU();
}

void* createScheme(const char* compModel, const char* gridModel, size_t type)
{
	if(type == CPU_SCHEME) {
		ComputationalScheme_CPU* scheme = new TestScheme_CPU();
    	return (void*) scheme;
	} else if(type == GPU_SCHEME) {
		ComputationalScheme_GPU* scheme;
		cudaMalloc((void**)&scheme, sizeof(TestScheme_GPU));
		create_GPUComputationalScheme<<<1, 1>>>(scheme);
		return (void*)scheme;
	} else {
		throw std::runtime_error("Unknown scheme type!");
	}
}

__device__ void LLLperformGPUSimulationStep(void* cu_field, void* cu_lr_halo,
        void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y)
{
	Cell* field = (Cell*)cu_field;
	int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
	field[tid_y * N_X + tid_x].r = 0.0;
}

__device__ void LLLupdateGPUGlobalBorders(void* cu_field, void* cu_lr_halo,
                void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y, size_t type)
{

}

__global__ void performGPUSimulationStep_wrapper(ComputationalScheme_GPU* scheme, void* cu_field, void* cu_lr_halo,
		void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y)
{
	LLLperformGPUSimulationStep(cu_field, cu_lr_halo, cu_tb_halo, cu_lrtb_halo, N_X, N_Y);
}

__global__ void updateGPUGlobalBorders_wrapper(ComputationalScheme_GPU* scheme, void* cu_field, void* cu_lr_halo,
		void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y, size_t type)
{
	LLLupdateGPUGlobalBorders(cu_field, cu_lr_halo, cu_tb_halo, cu_lrtb_halo, N_X, N_Y, type);
}

void performGPUSimulationStep_interface(void* scheme, void* cu_field, void* cu_lr_halo,
		void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y, size_t CUDA_X_BLOCKS, size_t CUDA_Y_BLOCKS,
		size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS, size_t SHAREDMEMORY, void* stream)
{
	cudaStream_t* cuStream = (cudaStream_t*)stream;
	performGPUSimulationStep_wrapper <<< dim3(CUDA_X_BLOCKS, CUDA_Y_BLOCKS, 1),
		dim3(CUDA_X_THREADS, CUDA_Y_THREADS, 1), SHAREDMEMORY,
		*cuStream >>> ((ComputationalScheme_GPU*)scheme, cu_field, cu_lr_halo, cu_tb_halo, cu_lrtb_halo, N_X, N_Y);
}

void updateGPUGlobalBorders_interface(void* scheme, void* cu_field, void* cu_lr_halo,
		void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y, size_t type, size_t CUDA_X_BLOCKS,
		size_t CUDA_Y_BLOCKS, size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS, size_t SHAREDMEMORY, void* stream)
{
	cudaStream_t* cuStream = (cudaStream_t*)stream;
	updateGPUGlobalBorders_wrapper <<< dim3(CUDA_X_BLOCKS, CUDA_Y_BLOCKS, 1),
		dim3(CUDA_X_THREADS, CUDA_Y_THREADS, 1), SHAREDMEMORY,
		*cuStream >>> ((ComputationalScheme_GPU*)scheme, cu_field, cu_lr_halo, cu_tb_halo, cu_lrtb_halo, N_X, N_Y, type);
}
