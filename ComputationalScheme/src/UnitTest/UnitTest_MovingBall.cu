#include "UnitTest_MovingBall.hpp"

typedef UnitTest_MovingBall::Cell Cell;

__global__ void performGPUSimulationStep_kernel(Cell* cu_field, Cell* cu_lr_halo,
		Cell* cu_tb_halo, Cell* cu_lrtb_halo, size_t N_X, size_t N_Y)
{
	size_t tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
	if(tid_x >= N_X || tid_y >= N_Y)
		return;
    STRUCT_DATA_TYPE data = 1000;
	Cell* C = nullptr;
    Cell* C1 = nullptr;
    C = &cu_field[tid_y * N_X + tid_x];
    if(tid_x == N_X-1) {
        data = cu_lr_halo[N_Y + tid_y].r;
    } else {
        C1 = &cu_field[tid_y * N_X + tid_x + 1];
        data = C1->r;
    }
    __syncthreads();
    C->r = data;
}

__global__ void updateGPUGlobalBorders_kernel(Cell* cu_field, Cell* cu_lr_halo,
		Cell* cu_tb_halo, Cell* cu_lrtb_halo, size_t N_X, size_t N_Y, size_t type)
{
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(type == CU_LEFT_BORDER) {
		cu_lr_halo[tid].r = cu_field[tid * N_X].r;
	} else if(type == CU_RIGHT_BORDER) {
        cu_lr_halo[N_Y + tid].r = 0.0;
	} else if(type == CU_TOP_BORDER) {
		cu_tb_halo[tid].r = 0.0;
	} else if(type == CU_BOTTOM_BORDER) {
		cu_tb_halo[N_Y + tid].r = 0.0;
	} else if(type == CU_LEFT_TOP_BORDER) {
		cu_lrtb_halo[CU_LEFT_TOP_BORDER - CU_LEFT_TOP_BORDER].r = 0.0;
	} else if(type == CU_RIGHT_TOP_BORDER) {
		cu_lrtb_halo[CU_RIGHT_TOP_BORDER - CU_LEFT_TOP_BORDER].r = 0.0;
	} else if(type == CU_LEFT_BOTTOM_BORDER) {
		cu_lrtb_halo[CU_LEFT_BOTTOM_BORDER - CU_LEFT_TOP_BORDER].r = 0.0;
	} else if(type == CU_RIGHT_BOTTOM_BORDER) {
		cu_lrtb_halo[CU_RIGHT_BOTTOM_BORDER - CU_LEFT_TOP_BORDER].r = 0.0;
	}
}

void* UnitTest_MovingBall::createField(size_t N_X, size_t N_Y)
{
    return (void*)(new Cell[N_X * N_Y]);
}

void* UnitTest_MovingBall::createPageLockedField(size_t N_X, size_t N_Y)
{
    Cell* ptr;
    HANDLE_CUERROR_PTR( cudaHostAlloc((void**)&ptr, N_X * N_Y * sizeof(Cell), cudaHostAllocDefault) );
    return (void*)ptr;
}

void* UnitTest_MovingBall::createGPUField(size_t N_X, size_t N_Y)
{
    Cell* ptr;
    HANDLE_CUERROR_PTR( cudaMalloc((void**)&ptr, N_X * N_Y * sizeof(Cell)) );
    return (void*)ptr;
}

ErrorStatus UnitTest_MovingBall::initField(void* field, size_t N_X, size_t N_Y)
{ // Initialization is on the CPU side
    Cell* cfield = (Cell*)field;
	Cell* C = nullptr;
    size_t global;
	float dx, dy;
    for(size_t x = 0; x < N_X; ++x) {
		dx = x / (float)N_X;
        for(size_t y = 0; y < N_Y; ++y) {
            global = y * N_X + x;
			C = &cfield[global];
			dy = y / (float)N_Y;
            /*if( (dy-0.5)*(dy-0.5) + (dx-0.75)*(dx-0.75) < 0.04 ) {
                C->r = 1.0;
            } else {
                C->r = 0.0;
            }*/
            if(x == 0 && y == 0)
                C->r = 0.0;
            else
                C->r = cos(dx) * cos(dy);
		}
    }
	return GPU_SUCCESS;
}

void* UnitTest_MovingBall::initHalos(size_t N)
{
    return (void*)(new Cell[N]);
}

void* UnitTest_MovingBall::initPageLockedHalos(size_t N)
{
    Cell* ptr;
    HANDLE_CUERROR_PTR( cudaHostAlloc((void**)&ptr, N * sizeof(Cell), cudaHostAllocDefault) );
    return (void*)ptr;
}

void* UnitTest_MovingBall::initGPUHalos(size_t N)
{
    Cell* ptr;
    HANDLE_CUERROR_PTR( cudaMalloc((void**)&ptr, N * sizeof(Cell)) );
    return (void*)ptr;
}

ErrorStatus UnitTest_MovingBall::performCPUSimulationStep(void* tmpCPUField, void* lr_halo,
        void* tb_halo, void* lrtb_halo, size_t N_X, size_t N_Y)
{
	return GPU_SUCCESS;
}

ErrorStatus UnitTest_MovingBall::performGPUSimulationStep(void* cu_field, void* cu_lr_halo,
        void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
		size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS, void* stream)
{
	size_t CUDA_X_BLOCKS = (size_t)((double)N_X / (double)CUDA_X_THREADS);
	size_t CUDA_Y_BLOCKS = (size_t)((double)N_Y / (double)CUDA_Y_THREADS);
	size_t SharedMemoryPerBlock = 0;
    cudaStream_t* cuStream = (cudaStream_t*)stream;
	/// Launch the CUDA kernel
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

ErrorStatus UnitTest_MovingBall::updateGPUGlobalBorders(void* cu_field, void* cu_lr_halo,
            void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
            size_t type, size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS, void* stream)
{
	static int k = 0;
	/// Calculate the amount of shared memory that is required for the kernel
	size_t sharedMemory = 0;
	cudaStream_t* cuStream = (cudaStream_t*)stream;
	size_t BLOCKS, THREADS;
	if(type == CU_TOP_BORDER || type == CU_BOTTOM_BORDER) {
		BLOCKS = (size_t)((double)N_X / (double)CUDA_X_THREADS);
		THREADS = CUDA_X_THREADS;
	}
	else if(type == CU_LEFT_BORDER || type == CU_RIGHT_BORDER) {
		BLOCKS = (size_t)((double)N_Y / (double)CUDA_Y_THREADS);
		THREADS = CUDA_Y_THREADS;
	} else {
		BLOCKS = 1;
		THREADS = 1;
	}
	/// Launch the CUDA kernel
	updateGPUGlobalBorders_kernel <<< dim3(BLOCKS, 1, 1),
		dim3(THREADS, 1, 1), sharedMemory,
		*cuStream >>> ((Cell*)cu_field, (Cell*)cu_lr_halo, (Cell*)cu_tb_halo,
            (Cell*)cu_lrtb_halo, N_X, N_Y, type);
	/// Check if the kernel executed without errors
	lastCudaError = cudaGetLastError();
	if(lastCudaError != cudaSuccess) {
		errorString = std::string("updateGPUGlobalBorders: ") +
			std::string(cudaGetErrorString(lastCudaError));
		return GPU_ERROR;
	}
	return GPU_SUCCESS;
}

void* UnitTest_MovingBall::getMarkerValue()
{
    return (void*)(&marker);
}

#ifdef __DEBUG__
ErrorStatus UnitTest_MovingBall::dbg_performSimulationStep(void* cu_field, void* cu_lr_halo,
	void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y, void* stream)
{
	size_t id = 0;
	Cell* C = (Cell*) cu_field;
	Cell* lr_halo = (Cell*) cu_lr_halo;
	for(size_t i = 0; i < N_X; ++i) {
		for(size_t j = 0; j < N_Y; ++j) {
			id = j * N_X + i;
			if(i == N_X - 1) {
				C[id] = lr_halo[N_Y + j];
			} else {
				C[id] = C[id + 1];
			}
		}
	}
	return GPU_SUCCESS;
}

ErrorStatus UnitTest_MovingBall::dbg_updateGlobalBorders(void* cu_field, void* cu_lr_halo,
	void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
	size_t type, size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS, void* stream)
{
	Cell* C = (Cell*) cu_field;
	Cell* lr_halo = (Cell*) cu_lr_halo;
	Cell* tb_halo = (Cell*) cu_tb_halo;
	Cell* lrtb_halo = (Cell*) cu_lrtb_halo;
	if(type == CU_LEFT_BORDER || type == CU_RIGHT_BORDER) {
		for(size_t j = 0; j < N_Y; ++j) {
			if(type == CU_LEFT_BORDER) {
				lr_halo[j].r = C[j * N_X].r;
			} else if(type == CU_RIGHT_BORDER) {
		        lr_halo[N_Y + j].r = 0.0;
			}
		}
	}
	if(type == CU_TOP_BORDER || type == CU_BOTTOM_BORDER) {
		for(size_t i = 0; i < N_X; ++i) {
			if(type == CU_TOP_BORDER) {
			   tb_halo[i].r = 0.0;
			} else if(type == CU_BOTTOM_BORDER) {
			   tb_halo[N_Y + i].r = 0.0;
			}
		}
	}
	if(type == CU_LEFT_TOP_BORDER) {
	   lrtb_halo[CU_LEFT_TOP_BORDER - CU_LEFT_TOP_BORDER].r = 0.0;
	} else if(type == CU_RIGHT_TOP_BORDER) {
	   lrtb_halo[CU_RIGHT_TOP_BORDER - CU_LEFT_TOP_BORDER].r = 0.0;
	} else if(type == CU_LEFT_BOTTOM_BORDER) {
	   lrtb_halo[CU_LEFT_BOTTOM_BORDER - CU_LEFT_TOP_BORDER].r = 0.0;
	} else if(type == CU_RIGHT_BOTTOM_BORDER) {
	   lrtb_halo[CU_RIGHT_BOTTOM_BORDER - CU_LEFT_TOP_BORDER].r = 0.0;
	}

	return GPU_SUCCESS;
}
#endif
