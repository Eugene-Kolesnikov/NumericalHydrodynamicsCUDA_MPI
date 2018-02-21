#include <ComputationalScheme/include/LBM/LatticeBoltzmannScheme.hpp>

typedef LatticeBoltzmannScheme::Cell Cell;

#include <string>
#include <exception>
#include <cmath>



/**
 * @brief
 */
__host__ __device__ void initLBParams(LBParams* p)
{
	STRUCT_DATA_TYPE W0 = 4.0 / 9.0;
	STRUCT_DATA_TYPE Wx = 1.0 / 9.0;
	STRUCT_DATA_TYPE Wxx = 1.0 / 36.0;
	p->Cs2 = 1.0 / 3.0;
	p->tau = 0.9;
	p->c[0] = make_float2(0.0f, 0.0f);	 p->w[0] = W0;
	p->c[1] = make_float2(1.0f, 0.0f);	 p->w[1] = Wx;
	p->c[2] = make_float2(-1.0f, 0.0f);	 p->w[2] = Wx;
	p->c[3] = make_float2(0.0f, 1.0f);	 p->w[3] = Wx;
	p->c[4] = make_float2(0.0f, -1.0f);	 p->w[4] = Wx;
	p->c[5] = make_float2(1.0f, 1.0f);	 p->w[5] = Wxx;
	p->c[6] = make_float2(1.0f, -1.0f);	 p->w[6] = Wxx;
	p->c[7] = make_float2(-1.0f, 1.0f);	 p->w[7] = Wxx;
	p->c[8] = make_float2(-1.0f, -1.0f); p->w[8] = Wxx;
}

/**
 * @brief
 */
__device__ void uploadTopBoundary(Cell* cu_field, Cell* cu_tb_halo, size_t N_X,
	size_t N_Y, size_t SHARED_X, size_t SHARED_Y)
{
	extern __shared__ Cell blockCells[];
	size_t tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t bid_x = threadIdx.x;
	if(tid_y == 0) {
		/// Global top border
		blockCells[bid_x + 1] = cu_tb_halo[bid_x];
	} else {
		blockCells[bid_x + 1] = cu_field[(tid_y - 1) * N_X + tid_x];
	}
}

/**
 * @brief
 */
__device__ void uploadBottomBoundary(Cell* cu_field, Cell* cu_tb_halo, size_t N_X,
	size_t N_Y, size_t SHARED_X, size_t SHARED_Y)
{
	extern __shared__ Cell blockCells[];
	size_t tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t bid_x = threadIdx.x;
	if(tid_y == N_Y - 1) {
		/// Global bottom border
		blockCells[(SHARED_Y - 1) * SHARED_X + bid_x + 1] = cu_tb_halo[N_X + bid_x];
	} else {
		blockCells[(SHARED_Y - 1) * SHARED_X + bid_x + 1] = cu_field[(tid_y + 1) * N_X + tid_x];
	}
}

/**
 * @brief
 */
__device__ void uploadLeftBoundary(Cell* cu_field, Cell* cu_lr_halo, size_t N_X,
	size_t N_Y, size_t SHARED_X, size_t SHARED_Y)
{
	extern __shared__ Cell blockCells[];
	size_t tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t bid_y = threadIdx.y;
	if(tid_x == 0) {
		/// Global left border
		blockCells[(bid_y + 1) * SHARED_X] = cu_lr_halo[bid_y];
	} else {
		blockCells[(bid_y + 1) * SHARED_X] = cu_field[tid_y * N_X + tid_x - 1];
	}
}

/**
 * @brief
 */
__device__ void uploadRightBoundary(Cell* cu_field, Cell* cu_lr_halo, size_t N_X,
	size_t N_Y, size_t SHARED_X, size_t SHARED_Y)
{
	extern __shared__ Cell blockCells[];
	size_t tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t bid_y = threadIdx.y;
	if(tid_x == N_X - 1) {
		/// Global right border
		blockCells[(bid_y + 2) * SHARED_X - 1] = cu_lr_halo[N_Y + bid_y];
	} else {
		blockCells[(bid_y + 2) * SHARED_X - 1] = cu_field[tid_y * N_X + tid_x + 1];
	}
}

/**
 * @brief
 */
__device__ void uploadDiagonalCells(Cell* cu_field, Cell* cu_tb_halo, Cell* cu_lr_halo,
	Cell* cu_lrtb_halo, size_t N_X, size_t N_Y, size_t SHARED_X, size_t SHARED_Y)
{
	extern __shared__ Cell blockCells[];
	size_t tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
	if(blockIdx.y == 0) {
		/// First row
		if(blockIdx.x == 0) {
			/// First cell of the first row
			blockCells[0] = cu_lrtb_halo[CU_LEFT_TOP_BORDER];
			blockCells[SHARED_X - 1] = cu_tb_halo[tid_x + SHARED_X];
			blockCells[(SHARED_Y - 1) * SHARED_X] = cu_lr_halo[tid_y + SHARED_Y];
			blockCells[SHARED_Y * SHARED_X - 1] = cu_field[(tid_y + SHARED_Y) * N_X + tid_x + SHARED_X];
		} else if(blockIdx.x == blockDim.x - 1) {
			/// Last cell of the first row
			blockCells[0] = cu_tb_halo[tid_x - 1];
			blockCells[SHARED_X - 1] = cu_lrtb_halo[CU_RIGHT_TOP_BORDER];;
			blockCells[(SHARED_Y - 1) * SHARED_X] = cu_field[(tid_y + SHARED_Y) * N_X + tid_x - 1];
			blockCells[SHARED_Y * SHARED_X - 1] = cu_lr_halo[N_Y + tid_y + SHARED_Y];
		} else {
			/// Internal cell of the first row
			blockCells[0] = cu_tb_halo[tid_x - 1];
			blockCells[SHARED_X - 1] = cu_tb_halo[tid_x + SHARED_X];
			blockCells[(SHARED_Y - 1) * SHARED_X] = cu_field[(tid_y + SHARED_Y) * N_X + tid_x - 1];
			blockCells[SHARED_Y * SHARED_X - 1] = cu_field[(tid_y + SHARED_Y) * N_X + tid_x + SHARED_X];
		}
	} else if(blockIdx.y == blockDim.y - 1) {
		/// Last row
		if(blockIdx.x == 0) {
			/// First cell of the last row
			blockCells[0] = cu_lr_halo[tid_y - 1];
			blockCells[SHARED_X - 1] = cu_field[(tid_y - 1) * N_X + tid_x + SHARED_X];
			blockCells[(SHARED_Y - 1) * SHARED_X] = cu_lrtb_halo[CU_LEFT_BOTTOM_BORDER];
			blockCells[SHARED_Y * SHARED_X - 1] = cu_tb_halo[N_X + tid_x + SHARED_X];
		} else if(blockIdx.x == blockDim.x - 1) {
			/// Last cell of the last row
			blockCells[0] = cu_field[(tid_y - 1) * N_X + tid_x-1];
			blockCells[SHARED_X - 1] = cu_lr_halo[N_Y + tid_y - 1];
			blockCells[(SHARED_Y - 1) * SHARED_X] = cu_tb_halo[N_X + tid_x - 1];
			blockCells[SHARED_Y * SHARED_X - 1] = cu_lrtb_halo[CU_RIGHT_BOTTOM_BORDER];
		} else {
			/// Internal cell of the last row
			blockCells[0] = cu_field[(tid_y - 1) * N_X + tid_x-1];
			blockCells[SHARED_X - 1] = cu_field[(tid_y - 1) * N_X + tid_x + SHARED_X];
			blockCells[(SHARED_Y - 1) * SHARED_X] = cu_tb_halo[N_X + tid_x - 1];
			blockCells[SHARED_Y * SHARED_X - 1] = cu_tb_halo[N_X + tid_x + SHARED_X];
		}
	} else {
		/// Internal cell of the grid
		blockCells[0] = cu_field[(tid_y - 1) * N_X + tid_x-1];
		blockCells[SHARED_X - 1] = cu_field[(tid_y - 1) * N_X + tid_x + SHARED_X];
		blockCells[(SHARED_Y - 1) * SHARED_X] = cu_field[(tid_y + SHARED_Y) * N_X + tid_x - 1];
		blockCells[SHARED_Y * SHARED_X - 1] = cu_field[(tid_y + SHARED_Y) * N_X + tid_x + SHARED_X];
	}
}

/**
 * @brief
 */
__device__ void initBlockCells(Cell* cu_field, Cell* cu_lr_halo, Cell* cu_tb_halo,
	Cell* cu_lrtb_halo, size_t N_X, size_t N_Y, size_t SHARED_X, size_t SHARED_Y)
{
	/// Size of the dynamic shared memory is ((CUDA_X_THREADS+2) * (CUDA_Y_THREADS+2)) to take into
	/// consideration the boundary values
	extern __shared__ Cell blockCells[];
	size_t tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t bid_x = threadIdx.x;
	size_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t bid_y = threadIdx.y;
	/// Upload the internal part of the cell grid
	blockCells[(bid_y + 1) * SHARED_X + bid_x + 1] = cu_field[tid_y * N_X + tid_x];
	if(bid_y == 0)
		uploadTopBoundary(cu_field, cu_tb_halo, N_X, N_Y, SHARED_X, SHARED_Y);
	if(bid_y == blockDim.y - 1)
		uploadBottomBoundary(cu_field, cu_tb_halo, N_X, N_Y, SHARED_X, SHARED_Y);
	if(bid_x == 0)
		uploadLeftBoundary(cu_field, cu_lr_halo, N_X, N_Y, SHARED_X, SHARED_Y);
	if(bid_x == blockDim.x - 1)
		uploadRightBoundary(cu_field, cu_lr_halo, N_X, N_Y, SHARED_X, SHARED_Y);
	if(bid_x == 0 && bid_y == 0)
		uploadDiagonalCells(cu_field, cu_tb_halo, cu_lr_halo, cu_lrtb_halo, N_X, N_Y, SHARED_X, SHARED_Y);
}

/**
 * @brief
 */
__device__ Cell* sCell(size_t x, size_t y)
{
	extern __shared__ Cell blockCells[];
	return &blockCells[(y + 1) * blockDim.x + x + 1];
}

/**
 * @brief
 */
__host__ __device__ float dot_float2(const float2& a1, const float2& a2)
{
	return a1.x * a2.x + a1.y * a2.y;
}

/**
 * @brief
 */
__device__ void streamingStep(Cell* C, LBParams* P)
{
	extern __shared__ Cell blockCells[];
	//size_t x = threadIdx.x;
	//size_t y = threadIdx.y;
	STRUCT_DATA_TYPE r = 0.0;
	//float2 u;
	//STRUCT_DATA_TYPE p = 0.0;
	/// Obtain values of mactoscopic parameters from populations of currounding cells
	/// Compute density of the cell
	for(size_t i = 0; i < DIRECTIONS_OF_INTERACTION; ++i) {
		r += C->F[i];
	}
	/*C->r = C->F0 + sCell(x+1,y)->Fmx + sCell(x-1,y)->Fx + sCell(x,y+1)->Fmy + sCell(x,y-1)->Fy +
		sCell(x-1,y-1)->Fxy + sCell(x+1,y-1)->Fmxy + sCell(x-1,y+1)->Fxmy + sCell(x+1,y+1)->Fmxmy;
	/// Compute velocity of the cell
	C->u = (P->c0.x * C->F0 + P->cmx.x * sCell(x+1,y)->Fmx + P->cx.x * sCell(x-1,y)->Fx +
		P->cmy.x * sCell(x,y+1)->Fmy + P->cy.x * sCell(x,y-1)->Fy + P->cxy.x * sCell(x-1,y-1)->Fxy +
		P->cmxy.x * sCell(x+1,y-1)->Fmxy + P->cxmy.x * sCell(x-1,y+1)->Fxmy +
		P->cmxmy.x * sCell(x+1,y+1)->Fmxmy) / C->r;
	C->v = (P->c0.y * C->F0 + P->cmx.y * sCell(x+1,y)->Fmx + P->cx.y * sCell(x-1,y)->Fx +
		P->cmy.y * sCell(x,y+1)->Fmy + P->cy.y * sCell(x,y-1)->Fy + P->cxy.y * sCell(x-1,y-1)->Fxy +
		P->cmxy.y * sCell(x+1,y-1)->Fmxy + P->cxmy.y * sCell(x-1,y+1)->Fxmy +
		P->cmxmy.y * sCell(x+1,y+1)->Fmxmy) / C->r;
	// Compute pressure of the cell
	C->p = dot_float2(P->c0, P->c0) * C->F0 + dot_float2(P->cmx, P->cmx) * sCell(x+1,y)->Fmx +
		dot_float2(P->cx, P->cx) * sCell(x-1,y)->Fx + dot_float2(P->cmy, P->cmy) * sCell(x,y+1)->Fmy +
		dot_float2(P->cy, P->cy) * sCell(x,y-1)->Fy + dot_float2(P->cxy, P->cxy) * sCell(x-1,y-1)->Fxy +
		dot_float2(P->cmxy, P->cmxy) * sCell(x+1,y-1)->Fmxy + dot_float2(P->cxmy, P->cxmy) * sCell(x-1,y+1)->Fxmy +
		dot_float2(P->cmxmy, P->cmxmy) * sCell(x+1,y+1)->Fmxmy;*/
}

/**
 * @brief
 */
__host__ __device__ STRUCT_DATA_TYPE computeFiEq(const STRUCT_DATA_TYPE& w, const STRUCT_DATA_TYPE& r, const float2 u,
	const float2 c, const STRUCT_DATA_TYPE& Cs2)
{
	STRUCT_DATA_TYPE dotCU = dot_float2(c, u);
	return w * r * (1 + dotCU / Cs2 + dotCU * dotCU / 2 / Cs2 / Cs2 - dot_float2(u, u) / 2 / Cs2);
}

/**
 * @brief
 */
__device__ void collisionStep(Cell* C, LBParams* P)
{
	float2 u = make_float2(C->u,C->v);
	/*C->F0 = C->F0 - (C->F0 - computeFiEq(P->W0, C->r, u, P->c0, P->Cs2)) / P->tau;
	C->Fx = C->Fx - (C->Fx - computeFiEq(P->Wx, C->r, u, P->cx, P->Cs2)) / P->tau;
	C->Fmx = C->Fmx - (C->Fmx - computeFiEq(P->Wx, C->r, u, P->cmx, P->Cs2)) / P->tau;
	C->Fy = C->Fy - (C->Fy - computeFiEq(P->Wx, C->r, u, P->cy, P->Cs2)) / P->tau;
	C->Fmy = C->Fmy - (C->Fmy - computeFiEq(P->Wx, C->r, u, P->cmy, P->Cs2)) / P->tau;
	C->Fxy = C->Fxy - (C->Fxy - computeFiEq(P->Wxx, C->r, u, P->cxy, P->Cs2)) / P->tau;
	C->Fmxy = C->Fmxy - (C->Fmxy - computeFiEq(P->Wxx, C->r, u, P->cmxy, P->Cs2)) / P->tau;
	C->Fxmy = C->Fxmy - (C->Fxmy - computeFiEq(P->Wxx, C->r, u, P->cxmy, P->Cs2)) / P->tau;
	C->Fmxmy = C->Fmxmy - (C->Fmxmy - computeFiEq(P->Wxx, C->r, u, P->cmxmy, P->Cs2)) / P->tau;*/
}

/**
 * @brief
 */
__global__ void performGPUSimulationStep_kernel(Cell* cu_field, Cell* cu_lr_halo,
		Cell* cu_tb_halo, Cell* cu_lrtb_halo, size_t N_X, size_t N_Y,
		size_t SHARED_X, size_t SHARED_Y)
{
	__shared__ LBParams P;
	size_t tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
	Cell* C = &cu_field[tid_y * N_X + tid_x];
	if(tid_x == 0 && tid_y == 0)
		initLBParams(&P);
	__syncthreads();
	/// Initialize the shared block Cells for faster memory accessible
	initBlockCells(cu_field, cu_lr_halo, cu_tb_halo, cu_lrtb_halo,
		N_X, N_Y, SHARED_X, SHARED_Y);
	__syncthreads();
	/// Streaming step (computation of local macroscopic parameters)
	streamingStep(C, &P);
	/// Synchronization of threads is not necessary since no interaction with
	/// other threads is performed
	/// Collision step (redistribution of the population)
	collisionStep(C, &P);
}

__device__ void initBlockCellsBorders(Cell* cu_field, size_t N_X, size_t N_Y, size_t type)
{
	extern __shared__ Cell blockCells[];
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t bid = threadIdx.x;
	if(type == CU_LEFT_BORDER) {
		blockCells[bid + 1] = cu_field[tid * N_X];
		/// Upload two border elements
		if(bid == 0) {
			/** The first left border element has a diagonal neighbor which
			 * is located in the top border. Since this interaction is not
			 * important for computations, we will use insted of the
			 * neighbor in the top border, the first element in the field
			 * (to fill in the space for easier logic)
			 */
			if(blockIdx.x != 0) {
				blockCells[0] = cu_field[(tid - 1) * N_X];
			} else {
				blockCells[0] = cu_field[tid * N_X];
			}
			/** The last left border element has a diagonal neighbor which
			 * is located in the bottom border. Since this interaction is not
			 * important for computations, we will use insted of the
			 * neighbor in the bottom border, the last element in the field
			 * (to fill in the space for easier logic)
			 */
			if(blockIdx.x != gridDim.x - 1) {
				blockCells[gridDim.x] = cu_field[(tid + gridDim.x) * N_X];
			} else {
				blockCells[gridDim.x] = cu_field[(tid + gridDim.x - 1) * N_X];
			}
		}
	} else if(type == CU_RIGHT_BORDER) {
		blockCells[bid] = cu_field[(tid + 1) * N_X - 1];
		/// Upload two border elements
		if(bid == 0) {
			/** The first left border element has a diagonal neighbor which
			 * is located in the top border. Since this interaction is not
			 * important for computations, we will use insted of the
			 * neighbor in the top border, the first element in the field
			 * (to fill in the space for easier logic)
			 */
			if(blockIdx.x != 0) {
				blockCells[0] = cu_field[tid * N_X - 1];
			} else {
				blockCells[0] = cu_field[(tid + 1) * N_X - 1];
			}
			/** The last left border element has a diagonal neighbor which
			 * is located in the bottom border. Since this interaction is not
			 * important for computations, we will use insted of the
			 * neighbor in the bottom border, the last element in the field
			 * (to fill in the space for easier logic)
			 */
			if(blockIdx.x != gridDim.x - 1) {
				blockCells[gridDim.x] = cu_field[(tid + gridDim.x + 1) * N_X - 1];
			} else {
				blockCells[gridDim.x] = cu_field[(tid + gridDim.x) * N_X - 1];
			}
		}
	} else if(type == CU_TOP_BORDER) {
		blockCells[bid] = cu_field[tid];
		/// Upload two border elements
		if(bid == 0) {
			if(blockIdx.x != 0) {
				blockCells[0] = cu_field[tid - 1];
			} else {
				blockCells[0] = cu_field[tid];
			}
			if(blockIdx.x != gridDim.x - 1) {
				blockCells[gridDim.x] = cu_field[tid + gridDim.x];
			} else {
				blockCells[gridDim.x] = cu_field[tid + gridDim.x - 1];
			}
		}
	} else if(type == CU_BOTTOM_BORDER) {
		blockCells[bid] = cu_field[(N_Y - 1) * N_X + tid];
		/// Upload two border elements
		if(bid == 0) {
			if(blockIdx.x != 0) {
				blockCells[0] = cu_field[(N_Y - 1) * N_X + tid - 1];
			} else {
				blockCells[0] = cu_field[(N_Y - 1) * N_X + tid];
			}
			if(blockIdx.x != gridDim.x - 1) {
				blockCells[gridDim.x] = cu_field[(N_Y - 1) * N_X + tid + gridDim.x];
			} else {
				blockCells[gridDim.x] = cu_field[(N_Y - 1) * N_X + tid + gridDim.x - 1];
			}
		}
	}
}

/**
 * @brief
 */
__global__ void updateGPUGlobalBorders_kernel(Cell* cu_field, Cell* cu_lr_halo,
		Cell* cu_tb_halo, Cell* cu_lrtb_halo, size_t N_X, size_t N_Y, size_t type)
{
	extern __shared__ Cell blockCells[];
	//size_t tid = threadIdx.x;
	initBlockCellsBorders(cu_field, N_X, N_Y, type);
	__syncthreads();
	/*if(type == CU_LEFT_BORDER) {
		cu_lr_halo[tid].Fx = blockCells[tid + 1].Fmx;
		cu_lr_halo[tid].Fxy = blockCells[tid].Fmxmy;
		cu_lr_halo[tid].Fxmy = blockCells[tid + 2].Fmxy;
	} else if(type == CU_RIGHT_BORDER) {
		cu_lr_halo[N_Y + tid].Fmx = blockCells[tid + 1].Fx;
		cu_lr_halo[N_Y + tid].Fmxmy = blockCells[tid].Fxy;
		cu_lr_halo[N_Y + tid].Fmxy = blockCells[tid + 2].Fxmy;
	} else if(type == CU_TOP_BORDER) {
		cu_tb_halo[tid].Fmy = blockCells[tid + 1].Fy;
		cu_tb_halo[tid].Fxmy = blockCells[tid].Fmxy;
		cu_tb_halo[tid].Fmxmy = blockCells[tid + 2].Fxy;
	} else if(type == CU_BOTTOM_BORDER) {
		cu_tb_halo[N_X + tid].Fy = blockCells[tid + 1].Fmy;
		cu_tb_halo[N_X + tid].Fmxy = blockCells[tid].Fxmy;
		cu_tb_halo[N_X + tid].Fxy = blockCells[tid + 2].Fmxmy;
	} else if(type == CU_LEFT_TOP_BORDER) {
		cu_lrtb_halo[LEFT_TOP_BORDER].Fxmy = cu_field[0].Fmxy;
	} else if(type == CU_RIGHT_TOP_BORDER) {
		cu_lrtb_halo[RIGHT_TOP_BORDER].Fmxmy = cu_field[N_X - 1].Fxy;
	} else if(type == CU_LEFT_BOTTOM_BORDER) {
		cu_lrtb_halo[LEFT_BOTTOM_BORDER].Fxy = cu_field[(N_Y - 1) * N_X].Fmxmy;
	} else if(type == CU_RIGHT_BOTTOM_BORDER) {
		cu_lrtb_halo[RIGHT_BOTTOM_BORDER].Fmxy = cu_field[N_Y * N_X - 1].Fxmy;
	}*/
}

void* LatticeBoltzmannScheme::createField(size_t N_X, size_t N_Y)
{
    return (void*)(new Cell[N_X * N_Y]);
}

void* LatticeBoltzmannScheme::createPageLockedField(size_t N_X, size_t N_Y)
{
    Cell* ptr;
    HANDLE_CUERROR_PTR( cudaHostAlloc((void**)&ptr, N_X * N_Y * sizeof(Cell), cudaHostAllocDefault) );
    return (void*)ptr;
}

void* LatticeBoltzmannScheme::createGPUField(size_t N_X, size_t N_Y)
{
    Cell* ptr;
    HANDLE_CUERROR_PTR( cudaMalloc((void**)&ptr, N_X * N_Y * sizeof(Cell)) );
    return (void*)ptr;
}

void* LatticeBoltzmannScheme::initHalos(size_t N)
{
    return (void*)(new Cell[N]);
}

void* LatticeBoltzmannScheme::initPageLockedHalos(size_t N)
{
    Cell* ptr;
    HANDLE_CUERROR_PTR( cudaHostAlloc((void**)&ptr, N * sizeof(Cell), cudaHostAllocDefault) );
    return (void*)ptr;
}

void* LatticeBoltzmannScheme::initGPUHalos(size_t N)
{
    Cell* ptr;
    HANDLE_CUERROR_PTR( cudaMalloc((void**)&ptr, N * sizeof(Cell)) );
    return (void*)ptr;
}

ErrorStatus LatticeBoltzmannScheme::performGPUSimulationStep(void* cu_field, void* cu_lr_halo,
        void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
		size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS, void* stream)
{
	return GPU_SUCCESS;
	/*size_t SHARED_X = CUDA_X_THREADS + 2;
	size_t SHARED_Y = CUDA_Y_THREADS + 2;
	size_t SharedMemoryPerBlock = SHARED_X * SHARED_Y * sizeof(Cell);
	float blocksPerSM = ceil((float)CUDA_X_BLOCKS * (float)CUDA_Y_BLOCKS / (float)amountSMs);
	size_t totalSharedMemoryPerBlock = ceil((float)totalSharedMemoryPerSM / blocksPerSM);*/
	/// Check if there is enough shared memory
	/*if(totalSharedMemoryPerBlock < SharedMemoryPerBlock) {
		errorString = std::string("Trying to allocate too much CUDA shared memory: ") +
			std::to_string(totalSharedMemoryPerBlock) + std::string(" bytes is available per block, ") +
			std::to_string(SharedMemoryPerBlock) + std::string(" bytes per block is requested!");
		return GPU_ERROR;
	}*/
    /*cudaStream_t* cuStream = (cudaStream_t*)stream;
	/// Launch the CUDA kernel
	performGPUSimulationStep_kernel <<< dim3(CUDA_X_BLOCKS, CUDA_Y_BLOCKS, 1),
		dim3(CUDA_X_THREADS, CUDA_Y_THREADS, 1), SharedMemoryPerBlock,
		*cuStream >>> ((Cell*)cu_field, (Cell*)cu_lr_halo, (Cell*)cu_tb_halo,
            (Cell*)cu_lrtb_halo, N_X, N_Y, SHARED_X, SHARED_Y);
	/// Check if the kernel executed without errors
	lastCudaError = cudaGetLastError();
	if(lastCudaError != cudaSuccess) {
		errorString = std::string("performGPUSimulationStep: ") +
			std::string(cudaGetErrorString(lastCudaError));
		return GPU_ERROR;
	}
	return GPU_SUCCESS;*/
}

ErrorStatus LatticeBoltzmannScheme::updateGPUGlobalBorders(void* cu_field, void* cu_lr_halo,
            void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
            size_t type, size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS, void* stream)
{
	return GPU_SUCCESS;
	/*/// Calculate the amount of shared memory that is required for the kernel
	size_t sharedMemory = 0;
	if(type == CU_LEFT_BORDER) {
		sharedMemory = (N_Y + 2) * sizeof(Cell);
	} else if(type == CU_RIGHT_BORDER) {
		sharedMemory = (N_Y + 2) * sizeof(Cell);
	} else if(type == CU_TOP_BORDER) {
		sharedMemory = (N_X + 2) * sizeof(Cell);
	} else if(type == CU_BOTTOM_BORDER) {
		sharedMemory = (N_X + 2) * sizeof(Cell);
	}
    cudaStream_t* cuStream = (cudaStream_t*)stream;
	/// Launch the CUDA kernel
	updateGPUGlobalBorders_kernel <<< dim3(CUDA_X_BLOCKS, CUDA_Y_BLOCKS, 1),
		dim3(CUDA_X_THREADS, CUDA_Y_THREADS, 1), sharedMemory,
		*cuStream >>> ((Cell*)cu_field, (Cell*)cu_lr_halo, (Cell*)cu_tb_halo,
            (Cell*)cu_lrtb_halo, N_X, N_Y, type);
	/// Check if the kernel executed without errors
	lastCudaError = cudaGetLastError();
	if(lastCudaError != cudaSuccess) {
		errorString = std::string("updateGPUGlobalBorders: ") +
			std::string(cudaGetErrorString(lastCudaError));
		return GPU_ERROR;
	}
	return GPU_SUCCESS;*/
}

void* LatticeBoltzmannScheme::getMarkerValue()
{
    return (void*)(&marker);
}


#ifdef __DEBUG__
ErrorStatus LatticeBoltzmannScheme::dbg_performSimulationStep(void* cu_field, void* cu_lr_halo,
    void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
	size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS,  void* stream)
{
	Cell* field = (Cell*)cu_field;
	Cell* lr_halo = (Cell*)cu_lr_halo;
	Cell* tb_halo = (Cell*)cu_tb_halo;
	Cell* lrtb_halo = (Cell*)cu_lrtb_halo;
	LBParams P;
	initLBParams(&P);
	dbg_streamingStep(field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y, &P);
	dbg_collisionStep(field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y, &P);
	return GPU_SUCCESS;
}

ErrorStatus LatticeBoltzmannScheme::dbg_updateGlobalBorders(void* cu_field, void* cu_lr_halo,
    void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
    size_t type, size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS, void* stream)
{
	Cell* field = (Cell*)cu_field;
	Cell* lr_halo = (Cell*)cu_lr_halo;
	Cell* tb_halo = (Cell*)cu_tb_halo;
	Cell* lrtb_halo = (Cell*)cu_lrtb_halo;
	size_t F_0 = 0, F_X = 1, F_mX = 2, F_Y = 3, F_mY = 4,
		F_XY = 5, F_XmY = 6, F_mXY = 7, F_mXmY = 8;
	if(type == CU_LEFT_BORDER) {
		for(int y = 0; y < N_Y; ++y) {
			lr_halo[y].F[F_X] = getCurCell(field, 0, y, N_X, N_Y)->F[F_mX];
			lr_halo[y].F[F_XY] = y != 0 ? getCurCell(field, 0, y-1, N_X, N_Y)->F[F_mXmY] : 0;
			lr_halo[y].F[F_XmY] = y != N_Y-1 ? getCurCell(field, 0, y+1, N_X, N_Y)->F[F_mXY] : 0;
		}
	} else if(type == CU_RIGHT_BORDER) {
		for(int y = 0; y < N_Y; ++y) {
			lr_halo[N_Y + y].F[F_mX] = getCurCell(field, N_X-1, y, N_X, N_Y)->F[F_X];
			lr_halo[N_Y + y].F[F_mXmY] = y != 0 ? getCurCell(field, N_X-1, y-1, N_X, N_Y)->F[F_XY] : 0;
			lr_halo[N_Y + y].F[F_mXY] = y != N_Y-1 ? getCurCell(field, N_X-1, y+1, N_X, N_Y)->F[F_XmY] : 0;
		}
	} else if(type == CU_TOP_BORDER) {
		for(int x = 0; x < N_X; ++x) {
			tb_halo[x].F[F_mY] = getCurCell(field, x, 0, N_X, N_Y)->F[F_Y];
			tb_halo[x].F[F_XmY] = x != N_X-1 ? getCurCell(field, x+1, 0, N_X, N_Y)->F[F_mXY] : 0;
			tb_halo[x].F[F_mXmY] = x != 0 ? getCurCell(field, x-1, 0, N_X, N_Y)->F[F_XY] : 0;
		}
	} else if(type == CU_BOTTOM_BORDER) {
		for(int x = 0; x < N_X; ++x) {
			tb_halo[N_X + x].F[F_Y] = getCurCell(field, x, N_Y-1, N_X, N_Y)->F[F_mY];
			tb_halo[N_X + x].F[F_XY] = x != N_X-1 ? getCurCell(field, x+1, N_Y-1, N_X, N_Y)->F[F_mXmY] : 0;
			tb_halo[N_X + x].F[F_mXY] = x != 0 ? getCurCell(field, x-1, N_Y-1, N_X, N_Y)->F[F_XmY] : 0;
		}
	} else if(type == CU_LEFT_TOP_BORDER) {
		type -= CU_LEFT_TOP_BORDER;
		lrtb_halo[type].F[F_XmY] = getCurCell(field, 0, 0, N_X, N_Y)->F[F_mXY];
	} else if(type == CU_RIGHT_TOP_BORDER) {
		type -= CU_LEFT_TOP_BORDER;
		lrtb_halo[type].F[F_mXmY] = getCurCell(field, N_X - 1, 0, N_X, N_Y)->F[F_XY];
	} else if(type == CU_LEFT_BOTTOM_BORDER) {
		type -= CU_LEFT_TOP_BORDER;
		lrtb_halo[type].F[F_XY] = getCurCell(field, 0, N_Y - 1, N_X, N_Y)->F[F_mXmY];
	} else if(type == CU_RIGHT_BOTTOM_BORDER) {
		type -= CU_LEFT_TOP_BORDER;
		lrtb_halo[type].F[F_mXY] = getCurCell(field, N_X - 1, N_Y - 1, N_X, N_Y)->F[F_XmY];
	}
	return GPU_SUCCESS;
}
#endif
