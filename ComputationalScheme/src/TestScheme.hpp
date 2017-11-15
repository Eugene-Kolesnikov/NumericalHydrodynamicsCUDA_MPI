#ifndef TESTSCHEME_HPP
#define TESTSCHEME_HPP

#include "GPUHeader.h"
#include "cell.h"
#include "ComputationalScheme.hpp"

class TestScheme : public ComputationalScheme {
public:
    TestScheme();
    virtual ~TestScheme(){}

public:
    virtual const std::type_info& getDataTypeid();
    virtual size_t getSizeOfDatatype();
    virtual size_t getNumberOfElements();
    virtual void* createField(size_t N_X, size_t N_Y);
    virtual void* createPageLockedField(size_t N_X, size_t N_Y);
    virtual void* createGPUField(size_t N_X, size_t N_Y);
    virtual ErrorStatus initField(void* field, size_t N_X, size_t N_Y);
    virtual void* initHalos(size_t N);
    virtual void* initPageLockedHalos(size_t N);
    virtual void* initGPUHalos(size_t N);
    virtual ErrorStatus performCPUSimulationStep(void* tmpCPUField, void* lr_halo,
        void* tb_halo, void* lrtb_halo, size_t N_X, size_t N_Y);
    virtual ErrorStatus performGPUSimulationStep(void* cu_field, void* cu_lr_halo,
            void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
            size_t CUDA_X_BLOCKS, size_t CUDA_Y_BLOCKS, size_t CUDA_X_THREADS,
            size_t CUDA_Y_THREADS, void* stream);
    virtual ErrorStatus updateGPUGlobalBorders(void* cu_field, void* cu_lr_halo,
                void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
                size_t type, size_t CUDA_X_BLOCKS, size_t CUDA_Y_BLOCKS,
                size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS, void* stream);
    virtual void* getMarkerValue();

protected:
    STRUCT_DATA_TYPE marker;
};


#endif // TESTSCHEME_HPP
