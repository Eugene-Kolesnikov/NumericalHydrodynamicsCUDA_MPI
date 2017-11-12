#ifndef TESTSCHEME_HPP
#define TESTSCHEME_HPP

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
    virtual void initField(void* field, size_t N_X, size_t N_Y);
    virtual void* initHalos(size_t N);
    virtual void* initPageLockedHalos(size_t N);
    virtual void* initGPUHalos(size_t N);
    virtual void performCPUSimulationStep(void* tmpCPUField, void* lr_halo, 
        void* tb_halo, size_t N_X, size_t N_Y);
    virtual void* getMarkerValue();
    
protected:
    STRUCT_DATA_TYPE marker;
};


#endif // TESTSCHEME_HPP