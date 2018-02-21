#ifndef LATTICEBOLTZMANNSCHEME_HPP
#define LATTICEBOLTZMANNSCHEME_HPP

#include <ComputationalScheme/include/GPUHeader.h>
#include <ComputationalScheme/include/CellConstruction.h>
#include <ComputationalScheme/include/ComputationalScheme.hpp>

#define STRUCT_DATA_TYPE double

#define DIRECTIONS_OF_INTERACTION 9

class LatticeBoltzmannScheme : public ComputationalScheme {
	GENERATE_CELL_STRUCTURE_WITH_SUPPORT_FUNCTIONS((r,1)(u,1)(v,1)(p,1)(t,1)(F,DIRECTIONS_OF_INTERACTION))
    //REGISTER_VISUALIZATION_PARAMETERS(("Density",0)("X-Velocity",1)("Y-Velocity",2)("Pressure",3))
    REGISTER_VISUALIZATION_PARAMETERS(("X-Velocity",1)("Y-Velocity",2))

public:
	LatticeBoltzmannScheme(): ComputationalScheme(){}
    virtual ~LatticeBoltzmannScheme(){}

public:
    virtual void* createField(size_t N_X, size_t N_Y) override;
    virtual void* createPageLockedField(size_t N_X, size_t N_Y) override;
    virtual void* createGPUField(size_t N_X, size_t N_Y) override;
    virtual ErrorStatus initField(void* field, size_t N_X, size_t N_Y,
		double X_MAX, double Y_MAX) override;
    virtual void* initHalos(size_t N) override;
    virtual void* initPageLockedHalos(size_t N) override;
    virtual void* initGPUHalos(size_t N) override;
    virtual ErrorStatus performCPUSimulationStep(void* tmpCPUField, void* lr_haloPtr,
		void* tb_haloPtr, void* lrtb_haloPtr, size_t N_X, size_t N_Y) override;
	virtual ErrorStatus updateCPUGlobalBorders(void* tmpCPUField, void* lr_haloPtr,
		void* tb_haloPtr, void* lrtb_haloPtr, size_t N_X, size_t N_Y, size_t type) override;
    virtual ErrorStatus performGPUSimulationStep(void* cu_field, void* cu_lr_halo,
            void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
			size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS, void* stream) override;
    virtual ErrorStatus updateGPUGlobalBorders(void* cu_field, void* cu_lr_halo,
                void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
                size_t type, size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS,
				void* stream) override;
    virtual void* getMarkerValue() override;

#ifdef __DEBUG__
    virtual ErrorStatus dbg_performSimulationStep(void* cu_field, void* cu_lr_halo,
        void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
		size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS,  void* stream) override;
	virtual ErrorStatus dbg_updateGlobalBorders(void* cu_field, void* cu_lr_halo,
        void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
        size_t type, size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS, void* stream) override;
#endif

protected:
    STRUCT_DATA_TYPE marker = -1;
};

struct LBParams
{
	STRUCT_DATA_TYPE w[DIRECTIONS_OF_INTERACTION];
	STRUCT_DATA_TYPE Cs2;
	STRUCT_DATA_TYPE tau;
	/**
	 * Vector 'c' is a vector of velocities for the families of particles
	 * in the cell. There are 9 different directions of movements:
	 * c(0,0), c(1,0), c(0,1), c(-1,0), c(0,-1), c(-1,-1), c(-1,1), c(1,-1), c(1,1)
	 */
	float2 c[DIRECTIONS_OF_INTERACTION];
	//STRUCT_DATA_TYPE uMax, Re, nu, omega;
	//STRUCT_DATA_TYPE obst_x, obst_y, obst_r;
	STRUCT_DATA_TYPE Re, d, k, uMax, nu, beta;
};


#endif // LATTICEBOLTZMANNSCHEME_HPP
