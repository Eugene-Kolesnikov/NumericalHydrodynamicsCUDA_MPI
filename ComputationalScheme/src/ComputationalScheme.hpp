#ifndef COMPUTATIONALSCHEME_HPP
#define COMPUTATIONALSCHEME_HPP

#include <typeinfo>
#include <cstdlib>
#include "GPUHeader.h"
#include <string>
#include <list>
#include <map>

#include "../../ComputationalModel/src/GPU_Status.h"

#define CU_LEFT_BORDER (0)
#define CU_RIGHT_BORDER (1)
#define CU_TOP_BORDER (2)
#define CU_BOTTOM_BORDER (3)
#define CU_LEFT_TOP_BORDER (4)
#define CU_RIGHT_TOP_BORDER (5)
#define CU_LEFT_BOTTOM_BORDER (6)
#define CU_RIGHT_BOTTOM_BORDER (7)

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
     * @brief
     * @return
     */
    virtual std::list<std::pair<std::string,size_t>> getDrawParams() = 0;

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
    virtual ErrorStatus initField(void* field, size_t N_X, size_t N_Y) = 0;

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
    virtual ErrorStatus performCPUSimulationStep(void* tmpCPUField, void* lr_halo,
            void* tb_halo, void* lrtb_halo, size_t N_X, size_t N_Y) = 0;

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
         size_t CUDA_X_BLOCKS, size_t CUDA_Y_BLOCKS, size_t CUDA_X_THREADS,
         size_t CUDA_Y_THREADS, void* stream) = 0;

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
         size_t type, size_t CUDA_X_BLOCKS, size_t CUDA_Y_BLOCKS,
         size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS, void* stream) = 0;

    /**
     * @brief
     * @return Pointer to the marker of type STRUCT_DATA_TYPE with an "unphysical"
     * value which indicates that the simulation has finished.
     */
    virtual void* getMarkerValue() = 0;

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
