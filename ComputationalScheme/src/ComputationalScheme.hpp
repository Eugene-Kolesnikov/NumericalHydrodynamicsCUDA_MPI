#ifndef COMPUTATIONALSCHEME_HPP
#define COMPUTATIONALSCHEME_HPP

#include <typeinfo>
#include <cstdlib>
#include "GPUHeader.h"

class ComputationalScheme {
public:
    ComputationalScheme(){};
    virtual ~ComputationalScheme(){};
    
    /**
     * @brief Both node types
     * @return return typeid(STRUCT_DATA_TYPE);
     */
    virtual const std::type_info& getDataTypeid() = 0;
    
    /**
     * @brief Both node types
     * @return sizeof(STRUCT_DATA_TYPE)
     */
    virtual size_t getSizeOfDatatype() = 0;
    
    /**
     * @brief 
     * @return static_cast<size_t>(sizeof(Cell) / sizeof(STRUCT_DATA_TYPE));
     */
    virtual size_t getNumberOfElements() = 0;
    
    /**
     * 
     * @param N_X
     * @param N_Y
     * @return Pointer to the created field casted to (void*).
     */
    virtual void* createField(size_t N_X, size_t N_Y) = 0;
    
    /**
     * 
     * @param N_X
     * @param N_Y
     * @return Pointer to the created field in the pinned memory casted to (void*).
     */
    virtual void* createPageLockedField(size_t N_X, size_t N_Y) = 0;
    
    /**
     * 
     * @param N_X
     * @param N_Y
     * @return Pointer to the created field in the GPU memory casted to (void*).
     */
    virtual void* createGPUField(size_t N_X, size_t N_Y) = 0;
    
    /**
     * @brief NODE_TYPE::SERVER_NODE
     * Initialize the global field
     * @param N_X
     * @param N_Y
     */
    virtual void initField(void* field, size_t N_X, size_t N_Y) = 0;
    
    /**
     * @brief 
     * @param N
     * @return Pointer to the created array of halo elements of size 2*N (since
     * halos must be created for both sides (either lefit-right or top-bottom)
     * at the same time) casted to (void*). 
     * new Cell[2*N]
     */
    virtual void* initHalos(size_t N) = 0;
    
    /**
     * @brief 
     * @param N
     * @return Pointer to the created array of halo elements of size 2*N (since
     * halos must be created for both sides (either lefit-right or top-bottom)
     * at the same time) in the pinned memory casted to (void*). 
     * new Cell[2*N]
     */
    virtual void* initPageLockedHalos(size_t N) = 0;
    
    /**
     * @brief 
     * @param N
     * @return Pointer to the created array of halo elements of size 2*N (since
     * halos must be created for both sides (either lefit-right or top-bottom)
     * at the same time) in the GPU memory casted to (void*). 
     * new Cell[2*N]
     */
    virtual void* initGPUHalos(size_t N) = 0;
    
    /**
     * @brief 
     * @param tmpCPUField
     * @param lr_halo
     * @param tb_halo
     * @param N_X
     * @param N_Y
     */
    virtual void performCPUSimulationStep(void* tmpCPUField, void* lr_halo, 
        void* tb_halo, size_t N_X, size_t N_Y) = 0;
    
    /**
     * @brief 
     * @return Pointer to the marker of type STRUCT_DATA_TYPE with an "unphysical"
     * value which indicates that the simulation has finished.
     */
    virtual void* getMarkerValue() = 0;
};

#endif // COMPUTATIONALSCHEME_HPP