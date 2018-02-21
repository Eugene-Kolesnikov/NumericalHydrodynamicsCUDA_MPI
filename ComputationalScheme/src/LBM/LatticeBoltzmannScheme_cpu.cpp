#include <ComputationalScheme/include/LBM/LatticeBoltzmannScheme.hpp>

typedef LatticeBoltzmannScheme::Cell Cell;

#include <string>
#include <exception>
#include <cmath>

#define OBSTACLE 0
#define FLUID 1

STRUCT_DATA_TYPE computeFeq(const STRUCT_DATA_TYPE& w, const STRUCT_DATA_TYPE& r, const float2 u,
	const float2 c, const STRUCT_DATA_TYPE& Cs2)
{
    STRUCT_DATA_TYPE cu = c.x * u.x +c.y * u.y;
    STRUCT_DATA_TYPE uu = u.x * u.x +u.y * u.y;
	//return w * r * (1 + cu / Cs2 + cu * cu / 2 / Cs2 / Cs2 - uu / 2 / Cs2);
    return r * w * (1 + 3*cu + 0.5 * 9*(cu * cu) - 3.0 / 2.0 * uu);
}

void initLBParams(LBParams* p, size_t N_X, size_t N_Y)
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
    /*p->obst_x = N_X / 5.0;      // position of the cylinder; the cylinder is
    p->obst_y = N_Y / 2.0;      // offset from the center to break symmetry
    p->obst_r = N_Y / 10.0 + 1.0;   // radius of the cylinder
    p->uMax = 0.2; // maximum velocity of the Poiseuille inflow
    p->Re = 100.0; // Reynolds number
    p->nu = p->uMax * 2.0 * p->obst_r / p->Re; // kinematic fluid viscosity
    p->omega = 1.0 / (3.0 * p->nu + 1.0 / 2.0); // relaxation parameter*/
    p->Re = 1000;
    p->d = 0.05;
    p->k = 80;
    p->uMax = 0.05; // maximum velocity
    p->nu = p->uMax * N_X / p->Re;
    p->beta = 1.0 / (3.0 * p->nu + 0.5); // relaxation parameter
}

/*// compute parabolic Poiseuille profile
double computePoiseuille(int iY, size_t N_Y, LBParams* p) {
    double y = (double)iY + 0.5;
    double L = (double)N_Y - 1;
    return 4.0 * p->uMax / (L*L) * (L*y - y*y);
}*/

ErrorStatus LatticeBoltzmannScheme::initField(void* field, size_t N_X, size_t N_Y,
    double X_MAX, double Y_MAX)
{ // Initialization is on the CPU side
	Cell* cfield = (Cell*)field;
	Cell* C = nullptr;
    size_t global;
	float2 u; // macroscopic (x,y)-velocity of the cell
	LBParams P;
	initLBParams(&P, N_X, N_Y);
    for(int x = 0; x < N_X; ++x) {
        for(int y = 0; y < N_Y; ++y) {
            global = y * N_X + x;
			C = &cfield[global];
            /*C->r = 1.0;
            C->u = computePoiseuille(y, N_Y, &P);
            C->v = 0.0;
            if ((x - P.obst_x) * (x - P.obst_x) + (y - P.obst_y) * (y - P.obst_y)
                <= P.obst_r * P.obst_r) {
                C->t = OBSTACLE;
            } else {
                C->t = FLUID;
            }
			u = make_float2(C->u,C->v);
			for(size_t i = 0; i < DIRECTIONS_OF_INTERACTION; ++i) {
				C->F[i] = computeFeq(P.w[i], C->r, u, P.c[i], P.Cs2);
			}
            if(C->t == OBSTACLE) {
                C->r = NAN;
                C->u = NAN;
                C->v = NAN;
            }*/
            C->r = 1.0;
            C->u = y <= ((double)N_Y / 2.0) ? P.uMax * tanh(P.k * ((double)y / (double)N_Y - 0.25))
                                            : P.uMax * tanh(P.k * (0.75 - (double)y / (double)N_Y));
            C->v = P.d * 1 * sin(2 * M_PI * ((double)x / (double)N_X + 0.25));
            C->t = FLUID;
            u = make_float2(C->u,C->v);
			for(size_t i = 0; i < DIRECTIONS_OF_INTERACTION; ++i) {
				C->F[i] = computeFeq(P.w[i], C->r, u, P.c[i], P.Cs2);
			}
		}
    }
	return GPU_SUCCESS;
}

Cell* getCellPtr(int x, int y, Cell* field, Cell* lr_halo, Cell* tb_halo,
	Cell* lrtb_halo, size_t N_X, size_t N_Y)
{
    size_t ltb = 0, rtb = 1, lbb = 2, rbb = 3;
	if(x >= 0) {
		if(x < N_X) {
			if(y < 0) { // top border element
				return &tb_halo[x];
			} else if(y == N_Y) { // bottom border element
				return &tb_halo[N_X + x];
			} else { // inside the field
				return &field[y * N_X + x];
			}
		} else { // right border + right-x diagonal elements
			if(y < 0) { // right_top diagonal element
				return &lrtb_halo[rtb];
			} else if(y == N_Y) { // right_bottom diagonal element
				return &lrtb_halo[rbb];
			} else { // right border element
				return &lr_halo[N_Y + y];
			}
		}
	} else {
		// left border + left-x diagonal elements
		if(y < 0) { // left_top diagonal element
			return &lrtb_halo[ltb];
		} else if(y == N_Y) { // left_bottom diagonal element
			return &lrtb_halo[lbb];
		} else { // left border element
			return &lr_halo[y];
		}
	}
}

Cell* getCell(int x, int y, size_t i, Cell* field, Cell* lr_halo, Cell* tb_halo,
	Cell* lrtb_halo, size_t N_X, size_t N_Y)
{
	if(i == 0) {
		return getCellPtr(x, y, field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y);
	} else if(i == 1) {
		return getCellPtr(x+1, y, field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y);
	} else if(i == 2) {
		return getCellPtr(x-1, y, field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y);
	} else if(i == 3) {
		return getCellPtr(x, y+1, field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y);
	} else if(i == 4) {
		return getCellPtr(x, y-1, field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y);
	} else if(i == 5) {
		return getCellPtr(x+1, y+1, field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y);
	} else if(i == 6) {
		return getCellPtr(x+1, y-1, field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y);
	} else if(i == 7) {
		return getCellPtr(x-1, y+1, field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y);
	} else { // if(i == 8) {
		return getCellPtr(x-1, y-1, field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y);
	}
}

Cell* getInvCell(int x, int y, size_t i, Cell* field, Cell* lr_halo, Cell* tb_halo,
	Cell* lrtb_halo, size_t N_X, size_t N_Y)
{
	if(i == 0) {
		return getCellPtr(x, y, field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y);
	} else if(i == 1) {
		return getCellPtr(x-1, y, field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y);
	} else if(i == 2) {
		return getCellPtr(x+1, y, field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y);
	} else if(i == 3) {
		return getCellPtr(x, y-1, field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y);
	} else if(i == 4) {
		return getCellPtr(x, y+1, field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y);
	} else if(i == 5) {
		return getCellPtr(x-1, y-1, field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y);
	} else if(i == 6) {
		return getCellPtr(x-1, y+1, field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y);
	} else if(i == 7) {
		return getCellPtr(x+1, y-1, field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y);
	} else { // if(i == 8) {
		return getCellPtr(x+1, y+1, field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y);
	}
}

size_t getInvId(size_t i)
{
    if(i == 0) {
		return 0;
	} else if(i == 1) {
		return 2;
	} else if(i == 2) {
		return 1;
	} else if(i == 3) {
		return 4;
	} else if(i == 4) {
		return 3;
	} else if(i == 5) {
		return 8;
	} else if(i == 6) {
		return 7;
	} else if(i == 7) {
		return 6;
	} else { // if(i == 8) {
		return 5;
	}
}

Cell* getCurCell(Cell* field, int x, int y, size_t N_X, size_t N_Y)
{
	return &field[y * N_X + x];
}

void cpu_streamingStep(Cell* field, Cell* lr_halo, Cell* tb_halo, Cell* lrtb_halo,
	size_t N_X, size_t N_Y, const LBParams* P)
{
	STRUCT_DATA_TYPE r = 0.0;
	float2 u = make_float2(0.0f, 0.0f);
	STRUCT_DATA_TYPE p = 0.0;
	float2 c = make_float2(0.0f, 0.0f);
	STRUCT_DATA_TYPE f = 0.0;
	Cell* C = nullptr;
	for(int x = 0; x < N_X; ++x) {
		for(int y = 0; y < N_Y; ++y) {
			r = 0.0;
			u = make_float2(0.0f, 0.0f);
			p = 0.0;
			C = getCurCell(field, x, y, N_X, N_Y);
            if(C->t == OBSTACLE)
                continue;
			/// Obtain values of mactoscopic parameters from populations of currounding cells
			for(size_t i = 0; i < DIRECTIONS_OF_INTERACTION; ++i) {
				c = P->c[i];
				f = getInvCell(x, y, i, field, lr_halo, tb_halo, lrtb_halo,
					 N_X, N_Y)->F[i];
				r += P->w[i] * f; /// Compute density of the cell
				u.x +=  P->w[i] * c.x * f; /// Compute (x,y)-velocity of the cell
                u.y +=  P->w[i] * c.y * f;
				p += (c.x * c.x + c.y * c.y) * f; /// Compute pressure of the cell
			}
			C->r = r;
			C->u = u.x / r;
			C->v = u.y / r;
			C->p = p;
		}
	}
}

void cpu_collisionStep(Cell* field, Cell* lr_halo, Cell* tb_halo, Cell* lrtb_halo,
	size_t N_X, size_t N_Y, const LBParams* P)
{
	Cell* C = nullptr;
    float2 u = make_float2(0.0f, 0.0f);
	for(int x = 0; x < N_X; ++x) {
		for(int y = 0; y < N_Y; ++y) {
			C = getCurCell(field, x, y, N_X, N_Y);
            if(C->t == OBSTACLE)
                continue;
            u = make_float2(C->u, C->v);
			for(size_t i = 0; i < DIRECTIONS_OF_INTERACTION; ++i) {
                C->F[i] += P->beta * (computeFeq(P->w[i], C->r, u, P->c[i], P->Cs2) - C->F[i]);
			}
		}
	}
}

void updateObstacles(Cell* field, Cell* lr_halo, Cell* tb_halo, Cell* lrtb_halo, size_t N_X, size_t N_Y, LBParams* P)
{
    Cell* C = nullptr;
    for(int x = 0; x < N_X; ++x) {
        for(int y = 0; y < N_Y; ++y) {
            C = getCurCell(field, x, y, N_X, N_Y);
            if(C->t == FLUID)
                continue;
            for(size_t i = 0; i < DIRECTIONS_OF_INTERACTION; ++i) {
                C->F[i] = getCell(x, y, i, field, lr_halo, tb_halo, lrtb_halo,
					 N_X, N_Y)->F[getInvId(i)];
            }
        }
    }
}

void updateObstaclesForVisualization(Cell* field, size_t N_X, size_t N_Y, LBParams* P)
{
    Cell* C = nullptr;
	for(int x = 0; x < N_X; ++x) {
		for(int y = 0; y < N_Y; ++y) {
			C = getCurCell(field, x, y, N_X, N_Y);
            if(C->t == FLUID)
                continue;
			for(size_t i = 0; i < DIRECTIONS_OF_INTERACTION; ++i) {
                C->r = NAN;
                C->u = NAN;
                C->v = NAN;
			}
		}
	}
}

ErrorStatus LatticeBoltzmannScheme::performCPUSimulationStep(void* tmpCPUField, void* lr_haloPtr,
	void* tb_haloPtr, void* lrtb_haloPtr, size_t N_X, size_t N_Y)
{
	Cell* field = (Cell*)tmpCPUField;
	Cell* lr_halo = (Cell*)lr_haloPtr;
	Cell* tb_halo = (Cell*)tb_haloPtr;
	Cell* lrtb_halo = (Cell*)lrtb_haloPtr;
	LBParams P;
	initLBParams(&P, N_X, N_Y);
    /*cpu_collisionStep(field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y, &P);
    updateObstacles(field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y, &P);
    cpu_streamingStep(field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y, &P);
    updateObstaclesForVisualization(field, N_X, N_Y, &P);*/
    cpu_collisionStep(field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y, &P);
    cpu_streamingStep(field, lr_halo, tb_halo, lrtb_halo, N_X, N_Y, &P);
	return GPU_SUCCESS;
}

ErrorStatus LatticeBoltzmannScheme::updateCPUGlobalBorders(void* tmpCPUField, void* lr_haloPtr,
	void* tb_haloPtr, void* lrtb_haloPtr, size_t N_X, size_t N_Y, size_t type)
{
    size_t ltb = 0, rtb = 1, lbb = 2, rbb = 3;
    //static bool init = false;
	Cell* field = (Cell*)tmpCPUField;
	Cell* lr_halo = (Cell*)lr_haloPtr;
	Cell* tb_halo = (Cell*)tb_haloPtr;
	Cell* lrtb_halo = (Cell*)lrtb_haloPtr;
	size_t F_0 = 0, F_X = 1, F_mX = 2, F_Y = 3, F_mY = 4,
		F_XY = 5, F_XmY = 6, F_mXY = 7, F_mXmY = 8;
    LBParams P;
	initLBParams(&P, N_X, N_Y);
    /*Cell* C;
    if(!init) {
        Cell* C1;
        Cell* C2;
        float2 u1, u2;
        for(int y = 0; y < N_Y; ++y) {
            C1 = &lr_halo[y];
            C2 = &lr_halo[N_Y + y];
            C1->r = 1.0;
            C2->r = 1.0;
            C1->u = computePoiseuille(y, N_Y, &P);
            C2->u = computePoiseuille(y, N_Y, &P);
            C1->v = 0.0;
            C2->v = 0.0;
            u1 = make_float2(C1->u,C1->v);
            u2 = make_float2(C2->u,C2->v);
            for(size_t i = 0; i < DIRECTIONS_OF_INTERACTION; ++i) {
                C1->F[i] = computeFeq(P.w[i], C1->r, u1, P.c[i], P.Cs2);
                C2->F[i] = computeFeq(P.w[i], C2->r, u2, P.c[i], P.Cs2);
            }
        }
        for(int x = 0; x < N_X; ++x) {
            C1 = &tb_halo[x];
            C2 = &tb_halo[N_X + x];
            C1->r = 1.0;
            C2->r = 1.0;
            C1->u = computePoiseuille(-1, N_Y, &P);
            C2->u = computePoiseuille(N_Y, N_Y, &P);
            C1->v = 0.0;
            C2->v = 0.0;
            u1 = make_float2(C1->u,C1->v);
            u2 = make_float2(C2->u,C2->v);
            for(size_t i = 0; i < DIRECTIONS_OF_INTERACTION; ++i) {
                C1->F[i] = computeFeq(P.w[i], C1->r, u1, P.c[i], P.Cs2);
                C2->F[i] = computeFeq(P.w[i], C2->r, u2, P.c[i], P.Cs2);
            }
        }
        for(size_t i = 0; i < 4; ++i) {
            C1 = &lrtb_halo[i];
            C1->r = 1.0;
            C1->v = 0.0;
            if(i == 0 || i == 1) { // top-left, top-right
                C1->u = computePoiseuille(-1, N_Y, &P);
            } else { // bottom-left, bottom-right
                C1->u = computePoiseuille(N_Y, N_Y, &P);
            }
            u1 = make_float2(C1->u,C1->v);
            for(size_t i = 0; i < DIRECTIONS_OF_INTERACTION; ++i) {
                C1->F[i] = computeFeq(P.w[i], C1->r, u1, P.c[i], P.Cs2);
            }
        }
        init = true;
    }*/
	if(type == CU_LEFT_BORDER) {
		for(int y = 0; y < N_Y; ++y) {
            /*C = &lr_halo[y];
            C->u = computePoiseuille(y, N_Y, &P);
            C->v = 0.0;
            C->r = 1.0 / (1.0 - C->u) * (C->F[F_0] + C->F[F_Y] + C->F[F_mY] +
                2 * (C->F[F_mX] + C->F[F_mXY] + C->F[F_mXmY]));
            C->F[F_X] = C->F[F_mX] + 2.0 / 3.0 * C->r * C->u;
            C->F[F_XY] = C->F[F_mXmY] + 0.5 * (C->F[F_mY] - C->F[F_Y]) +
                0.5 * C->r * C->v + 1.0 / 6.0 * C->r * C->u;
            C->F[F_XmY] = C->F[F_mXY] + 0.5 * (C->F[F_Y] - C->F[F_mY]) -
                0.5 * C->r * C->v + 1.0 / 6.0 * C->r * C->u;*/

            /*lr_halo[y].F[F_X] = getCurCell(field, 0, y, N_X, N_Y)->F[F_mX];
			lr_halo[y].F[F_XY] = y != 0 ? getCurCell(field, 0, y-1, N_X, N_Y)->F[F_mXmY] : 0;
			lr_halo[y].F[F_XmY] = y != N_Y-1 ? getCurCell(field, 0, y+1, N_X, N_Y)->F[F_mXY] : 0;*/

            lr_halo[y] = *getCurCell(field, N_X-1, y, N_X, N_Y);
		}
	} else if(type == CU_RIGHT_BORDER) {
		for(int y = 0; y < N_Y; ++y) {
            /*C = &lr_halo[N_Y + y];
            C->r = 1.0;
            C->u = -1.0 + 1.0 / C->r * (C->F[F_0] + C->F[F_Y] + C->F[F_mY] +
                2 * (C->F[F_X] + C->F[F_XmY] + C->F[F_XY]));
            C->v = 0.0;
            C->F[F_mX] = C->F[F_X] - 2.0 / 3.0 * C->r * C->u;
            C->F[F_mXY] = C->F[F_XmY] + 0.5 * (C->F[F_Y] - C->F[F_mY]) -
                0.5 * C->r * C->v - 1.0 / 6.0 * C->r * C->u;
            C->F[F_mXmY] = C->F[F_XY] + 0.5 * (C->F[F_mY] - C->F[F_Y]) +
                0.5 * C->r * C->v - 1.0 / 6.0 * C->r * C->u;*/

            /*lr_halo[N_Y + y].F[F_mX] = getCurCell(field, N_X-1, y, N_X, N_Y)->F[F_X];
			lr_halo[N_Y + y].F[F_mXmY] = y != 0 ? getCurCell(field, N_X-1, y-1, N_X, N_Y)->F[F_XY] : 0;
			lr_halo[N_Y + y].F[F_mXY] = y != N_Y-1 ? getCurCell(field, N_X-1, y+1, N_X, N_Y)->F[F_XmY] : 0;*/

            lr_halo[N_Y + y] = *getCurCell(field, 0, y, N_X, N_Y);
		}
	} else if(type == CU_TOP_BORDER) {
		for(int x = 0; x < N_X; ++x) {
			/*tb_halo[x].F[F_mY] = getCurCell(field, x, 0, N_X, N_Y)->F[F_Y];
			tb_halo[x].F[F_XmY] = x != N_X-1 ? getCurCell(field, x+1, 0, N_X, N_Y)->F[F_mXY] : 0;
			tb_halo[x].F[F_mXmY] = x != 0 ? getCurCell(field, x-1, 0, N_X, N_Y)->F[F_XY] : 0;*/
            tb_halo[x] = *getCurCell(field, x, N_Y-1, N_X, N_Y);
		}
	} else if(type == CU_BOTTOM_BORDER) {
		for(int x = 0; x < N_X; ++x) {
			/*tb_halo[N_X + x].F[F_Y] = getCurCell(field, x, N_Y-1, N_X, N_Y)->F[F_mY];
			tb_halo[N_X + x].F[F_XY] = x != N_X-1 ? getCurCell(field, x+1, N_Y-1, N_X, N_Y)->F[F_mXmY] : 0;
			tb_halo[N_X + x].F[F_mXY] = x != 0 ? getCurCell(field, x-1, N_Y-1, N_X, N_Y)->F[F_XmY] : 0;*/
            tb_halo[N_X + x] = *getCurCell(field, x, 0, N_X, N_Y);
		}
	} else if(type == CU_LEFT_TOP_BORDER) {
		type -= CU_LEFT_TOP_BORDER;
		/*lrtb_halo[type].F[F_XmY] = getCurCell(field, 0, 0, N_X, N_Y)->F[F_mXY];*/
        lrtb_halo[ltb] = *getCurCell(field, N_X-1, N_Y-1, N_X, N_Y);
	} else if(type == CU_RIGHT_TOP_BORDER) {
		type -= CU_LEFT_TOP_BORDER;
		/*lrtb_halo[type].F[F_mXmY] = getCurCell(field, N_X - 1, 0, N_X, N_Y)->F[F_XY];*/
        lrtb_halo[rtb] = *getCurCell(field, 0, N_Y-1, N_X, N_Y);
	} else if(type == CU_LEFT_BOTTOM_BORDER) {
		type -= CU_LEFT_TOP_BORDER;
		/*lrtb_halo[type].F[F_XY] = getCurCell(field, 0, N_Y - 1, N_X, N_Y)->F[F_mXmY];*/
        lrtb_halo[lbb] = *getCurCell(field, N_X-1, 0, N_X, N_Y);
	} else if(type == CU_RIGHT_BOTTOM_BORDER) {
		type -= CU_LEFT_TOP_BORDER;
		/*lrtb_halo[type].F[F_mXY] = getCurCell(field, N_X - 1, N_Y - 1, N_X, N_Y)->F[F_XmY];*/
        lrtb_halo[rbb] = *getCurCell(field, 0, 0, N_X, N_Y);
	}
	return GPU_SUCCESS;
}
