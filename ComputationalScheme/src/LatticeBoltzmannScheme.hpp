#ifndef LATTICEBOLTZMANNSCHEME_HPP
#define LATTICEBOLTZMANNSCHEME_HPP

#include "GPUHeader.h"
#include "LB_Cell.h"
#include "ComputationalScheme.hpp"

class LatticeBoltzmannScheme : public ComputationalScheme {
public:
	LatticeBoltzmannScheme(): ComputationalScheme(){}
    virtual ~LatticeBoltzmannScheme(){}

public:
    virtual const std::type_info& getDataTypeid() override;
    virtual size_t getSizeOfDatatype() override;
    virtual size_t getNumberOfElements() override;
	virtual std::list<std::pair<std::string,size_t>> getDrawParams() override;
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
            size_t CUDA_X_BLOCKS, size_t CUDA_Y_BLOCKS, size_t CUDA_X_THREADS,
            size_t CUDA_Y_THREADS, void* stream) override;
    virtual ErrorStatus updateGPUGlobalBorders(void* cu_field, void* cu_lr_halo,
                void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
                size_t type, size_t CUDA_X_BLOCKS, size_t CUDA_Y_BLOCKS,
                size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS, void* stream) override;
    virtual void* getMarkerValue() override;

protected:
    STRUCT_DATA_TYPE marker = -1;
};


#endif // LATTICEBOLTZMANNSCHEME_HPP
