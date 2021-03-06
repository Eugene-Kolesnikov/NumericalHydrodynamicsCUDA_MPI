#ifndef COMPUTATIONALSCHEME_HPP
#define COMPUTATIONALSCHEME_HPP

#include <typeinfo>
#include <cstdlib>
#include "GPUHeader.h"
#include <string>
#include <list>
#include <map>

#include <ComputationalModel/include/GPU_Status.h>
#include <Visualization/include/Visualizationproperty.hpp>

#define CU_LEFT_BORDER (0)
#define CU_RIGHT_BORDER (1)
#define CU_TOP_BORDER (2)
#define CU_BOTTOM_BORDER (3)
#define CU_LEFT_TOP_BORDER (4)
#define CU_RIGHT_TOP_BORDER (5)
#define CU_LEFT_BOTTOM_BORDER (6)
#define CU_RIGHT_BOTTOM_BORDER (7)

/*
    TODO: Create a function 'allocateField(enum memoryType = {CPU, CPU_PageLocked, GPU})'
    TODO: Create a function 'allocateArray(enum memoryType = {CPU, CPU_PageLocked, GPU})' for halos
    TODO: Change functions 'performCPUSimulationStep', 'performGPUSimulationStep',
        'updateCPUGlobalBorders', 'updateGPUGlobalBorders' to respective functions
        'performSimulationStep(enum SimulationType = {CPU, GPU})',
        'updateGlobalBorders(enum SimulationType = {CPU, GPU})'.
 */

class ComputationalScheme {
public:
    ComputationalScheme();
    virtual ~ComputationalScheme();

    /**
     * [initScheme description]
     */
    virtual ErrorStatus initScheme();

    /**
     * @brief Both node types
     * @return return typeid(STRUCT_DATA_TYPE);
     */
    virtual const std::type_info& getDataTypeid() const = 0;

    /**
     * @brief Both node types
     * @return sizeof(STRUCT_DATA_TYPE)
     */
    virtual size_t getSizeOfDatatype() const = 0;

    /**
     * @brief Both node types
     * @return sizeof(Cell)
     */
    virtual size_t getSizeOfDatastruct() const = 0;

    /**
     * @brief
     * @return number of elements in the Cell struct
     */
    virtual size_t getNumberOfElements() const = 0;

    /**
     * [getAmountOfArrayMembers description]
     * @return [description]
     */
    virtual const size_t* getAmountOfArrayMembers() = 0;

    /**
     * [getCellOffsets description]
     * @return [description]
     */
    virtual const size_t* getCellOffsets() = 0;

    /**
     * @brief
     * @return
     */
    virtual const std::vector<VisualizationProperty>* getDrawParams() = 0;

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
    virtual ErrorStatus initField(void* field, size_t N_X, size_t N_Y,
        double X_MAX, double Y_MAX) = 0;

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
     * @param lrtb_halo
     * @param N_X
     * @param N_Y
     */
    virtual ErrorStatus performCPUSimulationStep(void* tmpCPUField, void* lr_haloPtr,
    	void* tb_haloPtr, void* lrtb_haloPtr, size_t N_X, size_t N_Y) = 0;

    /**
     * [updateCPUGlobalBorders description]
     * @param  tmpCPUField [description]
     * @param  lr_halo     [description]
     * @param  tb_halo     [description]
     * @param  lrtb_halo   [description]
     * @param  N_X         [description]
     * @param  N_Y         [description]
     * @param  type        [description]
     * @return             [description]
     */
    virtual ErrorStatus updateCPUGlobalBorders(void* tmpCPUField, void* lr_haloPtr,
    	void* tb_haloPtr, void* lrtb_haloPtr, size_t N_X, size_t N_Y, size_t type) = 0;

    /**
     * [performGPUSimulationStep description]
     * @param cu_field       [description]
     * @param cu_lr_halo     [description]
     * @param cu_tb_halo     [description]
     * @param cu_lrtb_halo   [description]
     * @param N_X            [description]
     * @param N_Y            [description]
     * @param CUDA_X_BLOCKS  [description]
     * @param CUDA_Y_BLOCKS  [description]
     * @param CUDA_X_THREADS [description]
     * @param CUDA_Y_THREADS [description]
     * @param SHAREDMEMORY   [description]
     * @param stream         [description]
     */
     virtual ErrorStatus performGPUSimulationStep(void* cu_field, void* cu_lr_halo,
         void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
         size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS, void* stream) = 0;

    /**
     * [updateGPUGlobalBorders description]
     * @param cu_field       [description]
     * @param cu_lr_halo     [description]
     * @param cu_tb_halo     [description]
     * @param cu_lrtb_halo   [description]
     * @param N_X            [description]
     * @param N_Y            [description]
     * @param type           [description]
     * @param CUDA_X_BLOCKS  [description]
     * @param CUDA_Y_BLOCKS  [description]
     * @param CUDA_X_THREADS [description]
     * @param CUDA_Y_THREADS [description]
     * @param SHAREDMEMORY   [description]
     * @param stream         [description]
     */
     virtual ErrorStatus updateGPUGlobalBorders(void* cu_field, void* cu_lr_halo,
         void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
         size_t type, size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS, void* stream) = 0;

    /**
     * @brief
     * @return Pointer to the marker of type STRUCT_DATA_TYPE with an "unphysical"
     * value which indicates that the simulation has finished.
     */
    virtual void* getMarkerValue() = 0;

#ifdef __DEBUG__
    virtual ErrorStatus dbg_performSimulationStep(void* cu_field, void* cu_lr_halo,
        void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
        size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS, void* stream) = 0;
    virtual ErrorStatus dbg_updateGlobalBorders(void* cu_field, void* cu_lr_halo,
        void* cu_tb_halo, void* cu_lrtb_halo, size_t N_X, size_t N_Y,
        size_t type, size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS, void* stream) = 0;
#endif

public:
    /**
     * [getErrorString description]
     * @return [description]
     */
    std::string getErrorString();

private:
    ErrorStatus initDeviceProp();
    size_t getSheredMemoryPerSM(float arch);

protected:
    cudaError_t lastCudaError;
    cudaDeviceProp devProp;
    size_t amountSMs;
    size_t totalSharedMemoryPerSM;

protected:
    std::string errorString;
};

#endif // COMPUTATIONALSCHEME_HPP
