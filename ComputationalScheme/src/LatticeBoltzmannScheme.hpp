#ifndef LATTICEBOLTZMANNSCHEME_HPP
#define LATTICEBOLTZMANNSCHEME_HPP

#include "GPUHeader.h"
#include "CellConstruction.h"
#include "ComputationalScheme.hpp"

#define STRUCT_DATA_TYPE double

class LatticeBoltzmannScheme : public ComputationalScheme {
	GENERATE_CELL_STRUCTURE_WITH_SUPPORT_FUNCTIONS((r,1)(u,1)(v,1)(p,1)(F,9))
    REGISTER_VISUALIZATION_PARAMETERS(("Density",0)("X-Velocity",1)("Y-Velocity",2)("Pressure",3))

public:
	LatticeBoltzmannScheme(): ComputationalScheme(){}
    virtual ~LatticeBoltzmannScheme(){}

public:
    virtual void* createField(size_t N_X, size_t N_Y) override;
    virtual void* createPageLockedField(size_t N_X, size_t N_Y) override;
    virtual void* createGPUField(size_t N_X, size_t N_Y) override;
    virtual ErrorStatus initField(void* field, size_t N_X, size_t N_Y) override;
    virtual void* initHalos(size_t N) override;
    virtual void* initPageLockedHalos(size_t N) override;
    virtual void* initGPUHalos(size_t N) override;
    virtual ErrorStatus performCPUSimulationStep(void* tmpCPUField, void* lr_halo,
        void* tb_halo, void* lrtb_halo, size_t N_X, size_t N_Y) override;
    virtual ErrorStatus performGPUSimulationStep(void* cu_field, void* cu_lr_halo,
            void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
			size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS, void* stream) override;
    virtual ErrorStatus updateGPUGlobalBorders(void* cu_field, void* cu_lr_halo,
                void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
                size_t type, size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS,
				void* stream) override;
    virtual void* getMarkerValue() override;

protected:
    STRUCT_DATA_TYPE marker = -1;
};


#endif // LATTICEBOLTZMANNSCHEME_HPP
