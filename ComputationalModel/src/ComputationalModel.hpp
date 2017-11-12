/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   ComputationalModel.hpp
 * Author: eugene
 *
 * Created on November 1, 2017, 12:51 PM
 */

#ifndef COMPUTATIONALMODEL_HPP
#define COMPUTATIONALMODEL_HPP

#include <string>
#include <mpi.h>
#include "../../ComputationalScheme/src/ComputationalScheme.hpp"
#include "../../MpiController/src/FileLogger.hpp"
#include <dlfcn.h>

#define LEFT_BORDER (0)
#define RIGHT_BORDER (1)
#define TOP_BORDER (2)
#define BOTTOM_BORDER (3)

#define LEFT_TOP_BORDER (0)
#define RIGHT_TOP_BORDER (1)
#define LEFT_BOTTOM_BORDER (2)
#define RIGHT_BOTTOM_BORDER (3)

class ComputationalModel {
protected:
    /**
    * byte type is actually a char type. The reason to use it is that all other 
    * types like float, double, long double can be represented as a sequence of 
    * chars, which means that size (memory) of each type is a multiple of the char size.
    * It is useful to use byte since we don't know which one of the types (float, double,
    * long double) will be used in the program.
    */
   typedef char byte;
   
public:
    enum NODE_TYPE {SERVER_NODE, COMPUTATIONAL_NODE};
    ComputationalModel(const char* comp, const char* grid);
    virtual ~ComputationalModel();

/// Virtual functions which are unique for each computational model
public:
    /**
     * @brief This function is called by the ServerNode to update the global field
     * with values of a subfield which have been sent from a ComputationalNode with
     * 2D MPI ids (mpi_node_x, mpi_node_y) and are saved in a temporary array,
     * reference to which was obtained using the function
     * getTmpCPUFieldStoragePtr.
     * @param mpi_node_x -- X-coordinate of a 2D MPI ids of a ComputationalNode
     * from which the subfield was received.
     * @param mpi_node_y -- Y-coordinate of a 2D MPI ids of a ComputationalNode
     * from which the subfield was received.
     */
    virtual void updateGlobalField(size_t mpi_node_x, size_t mpi_node_y) = 0;
    
    /**
     * @brief The function is called by both types of nodes: ComputationalNode and
     * ServerNode when all necessary configuration variables are set. It allocates
     * memory on the CPU and GPU for all required computations.
     * For CPUComputationalModel only the following fields must be initialized:
     * field, tmpCPUField, lr_halo, tb_halo, lrtb_halo, rcv_lr_halo, rcv_tb_halo, 
     * rcv_lrtb_halo.
     * For GPUComputationalModel, besides the listed variables, also GPU specific
     * fields must be initialized. Moreover, allocated memory on the CPU must be 
     * page-locked and accessible to the device for asynchronous data transferring.
     */
    virtual void initializeEnvironment() = 0;

    /**
     * @brief Function which is called by both types of nodes: ComputationalNode and
     * ServerNode. ServerNode calls this function on the initialization step when
     * the global field must be divided into subfields for each ComputationalNode. In
     * this case, this function copies a subfield in the temporary storage array for
     * MPI data transferring to a ComputationalNode. ComputationalNode calls this
     * function when the visualization must be performed: it transfers the subfield
     * from the GPU memory to the temporary storage array in the CPU memory, which
     * is later used for transferring the subfield to the ServerNode.
     * @param mpi_node_x -- X-coordinate of the ComputationalNode, which is
     * obtained from the MPI_Node::globalMPI_id. This value is necessary only for
     * ServerNode calls since it must divide the global field into subfields
     * depending on their 2D MPI ids. In case when a ComputationalNode calls this
     * function, no arguments are submitted to the function and, therefore, by the
     * default rule, mpi_node_x = 0.
     * @param mpi_node_y -- Y-coordinate of the ComputationalNode, which is
     * obtained from the MPI_Node::globalMPI_id. This value is necessary only for
     * ServerNode calls since it must divide the global field into subfields
     * depending on their 2D MPI ids. In case when a ComputationalNode calls this
     * function, no arguments are submitted to the function and, therefore, by the
     * default rule, mpi_node_y = 0.
     */
    virtual void prepareSubfield(size_t mpi_node_x = 0, size_t mpi_node_y = 0) = 0;

    /**
     * @brief The function which is called only by ComputationalNode objects during the
     * initialization step after it receives its subfield which must be transferred
     * to the GPU memory.
     * Stream 'streamInternal' is responsible for this task.
     */
    virtual void loadSubFieldToGPU() = 0;

    /**
     * @brief This function is used to synchronize all GPU jobs that are in process
     * in the current moment, in particular, the purpose of this function is to
     * wait until all computations and CPU<->GPU memory transfers are finished.
     */
    virtual void gpuSync() = 0;

    /**
     * @brief The function is called by ComputationalNode objects to compute
     * internal elements of the subfield. Stream 'streamInternal' is responsible
     * for this task.
     */
    virtual void performSimulationStep() = 0;

    /**
     * @brief The function is called by ComputationalNode objects to transfer
     * halo elements to the GPU memory and update global borders depending on
     * the obtained subfield. Stream 'streamInternal' has to finish its task
     * prior to calling this function, since the correct values of the subfield
     * are essential for obtaining correct halo elements and global borders.
     * Stream 'streamHaloBorder' is responsible for this task.
     */
    virtual void updateHaloBorderElements() = 0;

    /**
     * @brief The function is called by ComputationalNode objects during the
     * ComputationalNode::runNode to transfer halo elements from the GPU memory
     * to the CPU memory for further transferring among ComputationalNode objects.
     * Stream 'streamHaloBorder' is responsible for this task.
     */
    virtual void prepareHaloElements() = 0;

/// Getters and setters for configuration parameters
public:
    void setLog(logging::FileLogger* _Log);
    logging::FileLogger* getLog() const;
    void setAppPath(std::string app_path);
    std::string getAppPath() const;
    void setNodeType(enum NODE_TYPE node_type);
    NODE_TYPE getNodeType() const;
    void setMPI_NODES_X(size_t val);
    size_t getMPI_NODES_X() const;
    void setMPI_NODES_Y(size_t val);
    size_t getMPI_NODES_Y() const;
    void setCUDA_X_THREADS(size_t val);
    size_t getCUDA_X_THREADS() const;
    void setCUDA_Y_THREADS(size_t val);
    size_t getCUDA_Y_THREADS() const;
    void setTAU(double val);
    double getTAU() const;
    void setN_X(size_t val);
    size_t getN_X() const;
    void setN_Y(size_t val);
    size_t getN_Y() const;
    void setLN_X(size_t val);
    size_t getLN_X() const;
    void setLN_Y(size_t val);
    size_t getLN_Y() const;
    
/// Functions which are the same for each computational model
public:
    /**
     * @brief Function which is called by both types of nodes: ComputationalNode and
     * ServerNode. This is a special function which is necessary for a correct
     * MPI transferring system, since the type of field cells is a complex structure.
     * It creates a special environment variable which determines how many bytes
     * of data is assigned to each field cell. It modifies the
     * MPI_CellType.
     */
    void createMpiStructType();
    
    /**
     * @brief The function is called by the ServerNode to initialize the field
     * at time t = 0.
     */
    void initializeField();
    
    /**
     * @brief Function which is called by both types of nodes: ComputationalNode and
     * ServerNode. It returns a pointer to an array located in the CPU memory
     * which is used to store subfield data of each ComputationalNode
     * (size: lN_X*lN_Y). This temporary storage is used by a ComputationalNode to
     * 1) obtain the initial subfield from the ServerNode which initializes the
     * whole field;
     * 2) to load the subfield from the GPU memory and later send it to the
     * ServerNode for further visualization.
     * and by the ServerNode to:
     * 1) collect a subfield from a ComputationalNode object to
     * update the global field for further visualization;
     * 2) to store a subfield of the global field, which must be sent to
     * a particular ComputationalNode.
     * @return Pointer to an array located in the CPU memory casted to (void*).
     */
    void* getTmpCPUFieldStoragePtr();
    
    /**
     * @brief The function which is called only by the ServerNode for the visualization,
     * since a pointer to the field is necessary for the visualization library (all
     * field elements are used in the visualization process).
     * @return Pointer to the field located in the CPU memory casted to (void*).
     */
    void* getField();
    
    /**
     * @brief The function is called by a ComputationalNode object to obtain a pointer
     * to an appropriate set of halos: right, left, top, or bottom for one of the
     * neighboring ComputationalNode objects. The values of the array referenced
     * by the pointer should be updated (correct values transferred from the GPU
     * memory) prior to calling this function. This pointer
     * will be used later for transferring the halo elements which are contained
     * in this array to another ComputationalNode object.
     * @param border_type -- required border: left, right, top, bottom.
     * @return Pointer to the array of halo points located in the CPU memory
     * casted to (void*).
     */
    void* getCPUHaloPtr(size_t border_type);
    
    /**
     * @brief The function is called by a ComputationalNode object to obtain a pointer
     * to an appropriate element of diagonal halos: left-top, right-top, left-bottom,
     * right-bottom for one of the neighboring ComputationalNode objects. The
     * value of the array referenced by the pointer should be updated (correct 
     * values transferred from the GPU memory) prior to calling this function. 
     * This pointer will be used later for transferring the diagonal halo element
     * which is contained in this array to another ComputationalNode object.
     * @param border_type -- required border: left-top, right-top, left-bottom,
     * right-bottom.
     * @return Pointer to the array of diagonal halo points located in the 
     * CPU memory casted to (void*).
     */
    void* getCPUDiagHaloPtr(size_t border_type);

    /**
     * @brief The function is called by a ComputationalNode object to obtain a
     * temporary pointer to a CPU memory which is used later as a destination
     * point during the MPI transfer of halo elements between the current
     * ComputationalNode object and another one.
     * @param border_type -- required border: left, right, top, bottom.
     * @return Pointer to the temporary array of halo points located in the CPU
     * memory casted to (void*).
     */
    void* getTmpCPUHaloPtr(size_t border_type);
    
    /**
     * @brief The function is called by a ComputationalNode object to obtain a
     * temporary pointer to a CPU memory which is used later as a destination
     * point during the MPI transfer of diagonal halo element between the current
     * ComputationalNode object and another one.
     * @param border_type -- required border: left-top, right-top, left-bottom,
     * right-bottom.
     * @return Pointer to the temporary array of diagonal halo points located in
     * the CPU memory casted to (void*).
     */
    void* getTmpCPUDiagHaloPtr(size_t border_type);

    /**
     * @brief The function is called only by the first ComputationalNode to set
     * up a marker which indicates that the simulation has been finished. This marker
     * should me set in the CPU subfield storage since no GPU->CPU data transferring
     * will be performed after this call. After this function is finished, the
     * subfield which is stored in the CPU memory will be sent to the ServerNode.
     * The ServerNode will check if the stop marker has been set and successfully
     * stop its work.
     */
    void setStopMarker();

    /**
     * @brief The function is called by the ServerNode to check if the stop marker
     * has been set. If the stop marker has been set, it indicates that the program
     * must finish its work.
     * @return True if the stop marker has been set and false otherwise.
     */
    bool checkStopMarker();
    
    /**
     * @brief Since each computational node has 4 extra rows/cols for correct
     * computations of subfield border elements, it is important to know if
     * the subfield border elements are global border elements or just halo points
     * which should be transferred among other ComputationalNode objects. For the
     * program to distinguish weather a row/col is a halo set or a global border,
     * array of 4 elements borders is implemented. If the
     * current ComputationalNode has a global border to the left of the subfield, then
     * borders[0] == 1; to the right -- borders[1] == 1; on the top --
     * borders[2] == 1; on the bottom -- borders[3] == 1. If a subfield of a
     * ComputationalNode doesn't have global borders, then all elements of
     * borders are zeros.
     * @param mpi_node_x -- X-coordinate of the ComputationalNode, which is
     * obtained from the MPI_Node::globalMPI_id
     * @param mpi_node_y -- Y-coordinate of the ComputationalNode, which is
     * obtained from the MPI_Node::globalMPI_id
     */
    void setBorders(size_t mpi_node_x, size_t mpi_node_y);
    
    /**
     * @brief 
     */
    void initScheme();

public:
    MPI_Datatype MPI_CellType;

protected:
    logging::FileLogger* Log;
    std::string appPath;
    NODE_TYPE nodeType;
    size_t MPI_NODES_X;
    size_t MPI_NODES_Y;
    size_t CUDA_X_THREADS;
    size_t CUDA_Y_THREADS;
    double TAU;
    size_t N_X;
    size_t N_Y;
    size_t lN_X;
    size_t lN_Y;
    size_t borders[4];
    std::string schemeModel;
    std::string gridModel;
    
protected:
    void* compSchemeLibHandle;
    void* (*createScheme)(const char* scheme, const char* grid);
    ComputationalScheme* scheme;
    
protected:
    void* field;
    void* tmpCPUField;
    void* lr_halo;
    void* tb_halo;
    void* lrtb_halo;
    void* rcv_lr_halo;
    void* rcv_tb_halo;
    void* rcv_lrtb_halo;
};

#endif /* COMPUTATIONALMODEL_HPP */
