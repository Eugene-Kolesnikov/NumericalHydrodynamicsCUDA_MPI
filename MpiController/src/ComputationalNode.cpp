/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   ProcessNode.cpp
 * Author: eugene
 *
 * Created on November 1, 2017, 12:49 PM
 */

#include "ComputationalNode.hpp"
#include "../../ComputationalModel/src/ComputationalModel.hpp" /// ComputationalModel::NODE_TYPE
#include <mpi.h>

ComputationalNode::ComputationalNode(size_t globalRank, size_t totalNodes, std::string app_path, int* _argc, char** _argv):
    MPI_Node(globalRank, totalNodes, app_path, _argc, _argv)
{
    compRanks = nullptr;
}

ComputationalNode::~ComputationalNode()
{
    if(compRanks != nullptr)
        delete[] compRanks;
    MPI_Group_free(&world_group);
    MPI_Group_free(&computational_group);
    MPI_Comm_free(&MPI_COMM_COMPUTATIONAL);
}

void ComputationalNode::initEnvironment()
{
    try {
        MPI_Node::initEnvironment();
        MPI_Node::setComputationalModelEnv(ComputationalModel::NODE_TYPE::COMPUTATIONAL_NODE);
        createComputationalMPIgroup();
        loadInitSubFieldFromServer();
        /// Transfer initial subfield from the CPU memory to the GPU memory
        /// Stream 'streamInternal' is responsible for this task.
        HANDLE_GPUERROR(model->loadSubFieldToGPU());
        /// Wait for the stream 'streamInternal' to finish its tasks
        HANDLE_GPUERROR(model->gpuSync());
        /// Transfer halo elements from the GPU memory to the CPU memory
        /// Stream 'streamHaloBorder' is responsible for this task.
        HANDLE_GPUERROR(model->prepareHaloElements());
        /// Wait for the stream 'streamHaloBorder' to finish its tasks
        HANDLE_GPUERROR(model->gpuSync());
        /// Share halo elements among ComputationalNode objects
        shareHaloElements();
        /// Transfer halo elements to the GPU memory and update global borders.
        /// Stream 'streamHaloBorder' is responsible for this task.
        HANDLE_GPUERROR(model->updateHaloBorderElements(localMPI_id_x,localMPI_id_y));
        /// Wait for the stream 'streamHaloBorder' to finish its tasks
        HANDLE_GPUERROR(model->gpuSync());
    } catch(const std::runtime_error& err) {
        if(Log.is_open())
            Log << _ERROR_ << std::string("(ComputationalNode:initEnvironment): ") + err.what();
        throw;
    }
}

void ComputationalNode::runNode()
{
    try {
        double curTime = 0.0; /// current time
        size_t perfSteps = 0; /// performed steps
        /// perform the computational loop from time 0 to TOTAL_TIME
        while(curTime < TOTAL_TIME)
        {
            Log << "Start calculations at time: " + std::to_string(curTime);
            /// Compute internal elements of the subfield.
            /// Stream 'streamInternal' is responsible for this task.
            HANDLE_GPUERROR(model->performSimulationStep());
            /// Wait for the stream 'streamInternal' to finish computations,
            /// since the correct values of the subfield are essential for
            /// obtaining correct halo elements and global borders.
            HANDLE_GPUERROR(model->gpuSync());
            /// Transfer halo elements from the GPU memory to the CPU memory
            /// Stream 'streamHaloBorder' is responsible for this task.
            HANDLE_GPUERROR(model->prepareHaloElements());
            /// Wait for the stream 'streamHaloBorder' to finish its tasks
            HANDLE_GPUERROR(model->gpuSync());
            /// Share halo elements among ComputationalNode objects
            shareHaloElements();
            /// Transfer halo elements to the GPU memory and update global borders.
            /// Stream 'streamHaloBorder' is responsible for this task.
            HANDLE_GPUERROR(model->updateHaloBorderElements(localMPI_id_x, localMPI_id_y));
            ++perfSteps;
            if(perfSteps == STEP_LENGTH) {
                /** Request to transfer the subfield from the GPU memory to the
                 * CPU memory.
                 * Stream 'streamInternal' is used for this task.
                 * Since tasks of streams 'streamInternal' and 'streamHaloBorder'
                 * don't intersect, the synchronization is not needed.
                 */
                HANDLE_GPUERROR(model->prepareSubfield());
                /// Wait for the stream 'streamHaloBorder' and 'streamInternal'
                /// to finish their tasks
                HANDLE_GPUERROR(model->gpuSync());
                sendUpdatedSubFieldToServer();
                perfSteps = 0;
            } else {
                /// Wait for the stream 'streamHaloBorder' to finish its tasks
                HANDLE_GPUERROR(model->gpuSync());
            }
            curTime += TAU;
        }
        setStopMarker();
        sendUpdatedSubFieldToServer();
        MPI_Node::finalBarrierSync();
        model->deinitModel();
    } catch(const std::runtime_error& err) {
        Log << _ERROR_ << std::string("(ComputationalNode:runNode): ") + err.what();
        throw;
    }
}

void ComputationalNode::createComputationalMPIgroup()
{
    /// Get the group of processes in MPI_COMM_WORLD
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    /** Construct a group containing all of the ranks which refer to
     * computational nodes in world_group.
     * They are all except the last one, which is a server node.
     */
    int numberOfRanks = totalMPINodes - 1;
    compRanks = new int[numberOfRanks];
    for(int i = 0; i < numberOfRanks; ++i)
        compRanks[i] = i;
    MPI_Group_incl(world_group, numberOfRanks, compRanks, &computational_group);

    int tag = numberOfRanks; /// safe tag unused by other communication
    /// Create a new communicator based on the group
    MPI_Comm_create_group(MPI_COMM_WORLD, computational_group, tag, &MPI_COMM_COMPUTATIONAL);
}

void ComputationalNode::loadInitSubFieldFromServer()
{
    Log << "Uploading the initialized subfield";
    MPI_Status status;
    size_t serverProcess_id = totalMPINodes - 1;
    size_t totalAmountCellsToTransfer = lN_X * lN_Y;
    Log << "Sending a request for a temporary CPU field storage pointer";
    void* tmpStoragePtr = model->getTmpCPUFieldStoragePtr();
    Log << "Received the temporary CPU field storage pointer";
    Log << "Trying to receive " +
            std::to_string(totalAmountCellsToTransfer) +
            " amount of field points from the server node";
    MPI_Recv(tmpStoragePtr, totalAmountCellsToTransfer,
            model->MPI_CellType, serverProcess_id,
            globalMPI_id, MPI_COMM_WORLD, &status);
    /// After receiving the message, check the status to determine
    /// how many numbers were actually received
    int number_amount;
    MPI_Get_count(&status, model->MPI_CellType, &number_amount);
    if(number_amount != totalAmountCellsToTransfer) {
        Log << _WARNING_ << ("Received " + std::to_string(number_amount) +
                " amount of field cells instead of " + std::to_string(totalAmountCellsToTransfer));
    }
    Log << "Successfully received data from the server node";
    MPI_Barrier(MPI_COMM_WORLD);
    Log << "Barrier synchronization has been successfully performed";
}

void ComputationalNode::sendUpdatedSubFieldToServer()
{
    int mpi_err_status, resultlen;
    char err_buffer[MPI_MAX_ERROR_STRING];
    Log << "Sharing the updated subfield";
    size_t totalAmountCellsToTransfer = lN_X * lN_Y;
    size_t globalMPIidServer = totalMPINodes - 1;
    Log << "Sending a request for a temporary CPU field storage pointer";
    void* tmpStoragePtr = model->getTmpCPUFieldStoragePtr();
    Log << "Received the temporary CPU field storage pointer";
    Log << "Trying to send " + std::to_string(totalAmountCellsToTransfer) +
            " amount of data to server node.";
    mpi_err_status = MPI_Send(tmpStoragePtr, totalAmountCellsToTransfer,
            model->MPI_CellType, globalMPIidServer, globalMPI_id, MPI_COMM_WORLD);
    if(mpi_err_status != MPI_SUCCESS) {
        MPI_Error_string(mpi_err_status, err_buffer, &resultlen);
        throw std::runtime_error(err_buffer);
    }
    Log << "Data has been successfully sent to the server node";
    /// make sure that Server node has successfully received all subfields
    MPI_Barrier(MPI_COMM_WORLD);
    Log << "Barrier synchronization has been successfully performed";
}

void ComputationalNode::shareHaloElements()
{
    /** The nodes are represented not as geometrical objects in the Cartesian
     * coordinates where Y-vector is heading upwards, but as a matrix
     * where (i+1)-th row is located beneath the i-th row. Therefore,
     * if the current node has MPI coordinates (x,y), then the top neighbor
     * will have MPI coordinates (x,y-1); the bottom neighbor -- (x,y+1).
     * If the node doesn't have a top neighbor, then the id for the top
     * neighbor is MPI_PROC_NULL. The same rule is for other neighbors.
     */
    int left_neighbor_id =
        (localMPI_id_x == 0) ? MPI_PROC_NULL :
            getGlobalMPIid(localMPI_id_x - 1, localMPI_id_y);
    int right_neighbor_id =
        (localMPI_id_x == (MPI_NODES_X - 1)) ? MPI_PROC_NULL :
            getGlobalMPIid(localMPI_id_x + 1, localMPI_id_y);
    int top_neighbor_id =
        (localMPI_id_y == 0) ? MPI_PROC_NULL :
            getGlobalMPIid(localMPI_id_x, localMPI_id_y - 1);
    int bottom_neighbor_id =
        (localMPI_id_y == (MPI_NODES_Y - 1)) ? MPI_PROC_NULL :
            getGlobalMPIid(localMPI_id_x, localMPI_id_y + 1);
    /// Sending left halo elements to the left neighbor and at the same time,
    /// receiving right halo elements from the right neighbor
    sndRcvHaloElements(left_neighbor_id, right_neighbor_id, LEFT_BORDER, RIGHT_BORDER);
    /// Sending right halo elements to the right neighbor and at the same time,
    /// receiving left halo elements from the left neighbor
    sndRcvHaloElements(right_neighbor_id, left_neighbor_id, RIGHT_BORDER, LEFT_BORDER);
    /// Sending top halo elements to the top neighbor and at the same time,
    /// receiving bottom halo elements from the bottom neighbor
    sndRcvHaloElements(top_neighbor_id, bottom_neighbor_id, TOP_BORDER, BOTTOM_BORDER);
    /// Sending bottom halo elements to the bottom neighbor and at the same time,
    /// receiving top halo elements from the top neighbor
    sndRcvHaloElements(bottom_neighbor_id, top_neighbor_id, BOTTOM_BORDER, TOP_BORDER);

    /// Sending and receiving diagonal halo elements
    int left_top_neighbor_id =
        (localMPI_id_x == 0 || localMPI_id_y == 0) ? MPI_PROC_NULL :
            getGlobalMPIid(localMPI_id_x - 1, localMPI_id_y - 1);
    int right_top_neighbor_id =
        (localMPI_id_x == (MPI_NODES_X - 1) || localMPI_id_y == 0) ? MPI_PROC_NULL :
            getGlobalMPIid(localMPI_id_x + 1, localMPI_id_y - 1);
    int left_bottom_neighbor_id =
        (localMPI_id_x == 0 || localMPI_id_y == (MPI_NODES_Y - 1)) ? MPI_PROC_NULL :
            getGlobalMPIid(localMPI_id_x - 1, localMPI_id_y + 1);
    int right_bottom_neighbor_id =
        (localMPI_id_x == (MPI_NODES_X - 1) || localMPI_id_y == (MPI_NODES_Y - 1)) ? MPI_PROC_NULL :
            getGlobalMPIid(localMPI_id_x + 1, localMPI_id_y + 1);
    /// Sending left-top halo element to the left-top neighbor and at the same time,
    /// receiving right-bottom halo element from the right-bottom neighbor
    sndRcvDiagHaloElements(left_top_neighbor_id, right_bottom_neighbor_id,
            LEFT_TOP_BORDER, RIGHT_BOTTOM_BORDER);
    /// Sending right-bottom halo element to the right-bottom neighbor and at the same time,
    /// receiving left-top halo element from the left-top neighbor
    sndRcvDiagHaloElements(right_bottom_neighbor_id, left_top_neighbor_id,
            RIGHT_BOTTOM_BORDER, LEFT_TOP_BORDER);
    /// Sending right-top halo element to the right-top neighbor and at the same time,
    /// receiving left-bottom halo element from the left-bottom neighbor
    sndRcvDiagHaloElements(right_top_neighbor_id, left_bottom_neighbor_id,
            RIGHT_TOP_BORDER, LEFT_BOTTOM_BORDER);
    /// Sending left-bottom halo element to the left-bottom neighbor and at the same time,
    /// receiving right-top halo element from the right-top neighbor
    sndRcvDiagHaloElements(left_bottom_neighbor_id, right_top_neighbor_id,
            LEFT_BOTTOM_BORDER, RIGHT_TOP_BORDER);
}

void ComputationalNode::sndRcvHaloElements(int snd_id, int rcv_id, int snd_border, int rcv_border)
{
    MPI_Status status;
    int mpi_err_status, resultlen;
    char err_buffer[MPI_MAX_ERROR_STRING];
    size_t sndLocalIdx, sndLocalIdy, rcvLocalIdx, rcvLocalIdy;
    size_t AmountCellsToTransfer;
    if(snd_border == LEFT_BORDER || snd_border == RIGHT_BORDER) {
        AmountCellsToTransfer = lN_Y;
    } else {
        AmountCellsToTransfer = lN_X;
    }
    setLocalMPI_ids(snd_id, sndLocalIdx, sndLocalIdy);
    setLocalMPI_ids(rcv_id, rcvLocalIdx, rcvLocalIdy);
    Log << "Trying to send " + std::to_string(AmountCellsToTransfer) +
         " amount of field cells to node (" + std::to_string((int)sndLocalIdx) +
         "," + std::to_string((int)sndLocalIdy) + " | global: " +
         std::to_string(snd_id) + ") and to receive " +
         std::to_string(AmountCellsToTransfer) + " amount of field cells from node (" +
         std::to_string((int)rcvLocalIdx) + "," + std::to_string((int)rcvLocalIdy) +
         " | global: " + std::to_string(rcv_id) + ").";
    /// Obtain a pointer to the array with halo points for the neighboring
    /// ComputationalNode which were previously loaded from the GPU memory.
    void* sndHaloPtr = model->getCPUHaloPtr(snd_border);
    /** Obtain a pointer to the temporary array of halo points for the current
     * ComputationalNode. After data transferring they will replace halo points
     * which are stored in the GPU memory.
     */
    void* rcvHaloPtr = model->getTmpCPUHaloPtr(rcv_border);
    /// Sending halo elements to the snd_id ComputationalNode and at the same time,
    /// receiving halo elements from the rcv_id ComputationalNode.
    mpi_err_status = MPI_Sendrecv(sndHaloPtr, AmountCellsToTransfer, model->MPI_CellType,
            snd_id, globalMPI_id, rcvHaloPtr, AmountCellsToTransfer, model->MPI_CellType,
            rcv_id, rcv_id, MPI_COMM_WORLD, &status);
    /// Check if the MPI transfer was successful
    if(mpi_err_status != MPI_SUCCESS) {
        MPI_Error_string(mpi_err_status, err_buffer, &resultlen);
        throw std::runtime_error(err_buffer);
    }
    /// After receiving the message, check the status to determine
    /// how many cells were actually received
    int number_amount;
    MPI_Get_count(&status, model->MPI_CellType, &number_amount);
    if(number_amount != AmountCellsToTransfer) {
        Log << _WARNING_ << ("Received " + std::to_string(number_amount) +
                " amount of field cells instead of " + std::to_string(AmountCellsToTransfer));
    }
    Log << "Data has been successfully sent and received.";
    /// Make sure that every ComputationalNode sent and received halo elements
    mpi_err_status = MPI_Barrier(MPI_COMM_COMPUTATIONAL);
    /// Check if the MPI barrier synchronization was successful
    if(mpi_err_status != MPI_SUCCESS) {
        MPI_Error_string(mpi_err_status, err_buffer, &resultlen);
        throw std::runtime_error(err_buffer);
    }
    Log << "Barrier synchronization has been successfully performed";
}

void ComputationalNode::sndRcvDiagHaloElements(int snd_id, int rcv_id, int snd_border, int rcv_border)
{
    MPI_Status status;
    int mpi_err_status, resultlen;
    char err_buffer[MPI_MAX_ERROR_STRING];
    size_t sndLocalIdx, sndLocalIdy, rcvLocalIdx, rcvLocalIdy;
    size_t AmountCellsToTransfer = 1;
    setLocalMPI_ids(snd_id, sndLocalIdx, sndLocalIdy);
    setLocalMPI_ids(rcv_id, rcvLocalIdx, rcvLocalIdy);
    Log << "Trying to send " + std::to_string(AmountCellsToTransfer) +
         " amount of diagonal field cells to node (" + std::to_string((int)sndLocalIdx) +
         "," + std::to_string((int)sndLocalIdy) + " | global: " +
         std::to_string(snd_id) + ") and to receive " +
         std::to_string(AmountCellsToTransfer) + " amount of diagonal field cells from node (" +
         std::to_string((int)rcvLocalIdx) + "," + std::to_string((int)rcvLocalIdy) +
         " | global: " + std::to_string(rcv_id) + ").";
    /// Obtain a pointer to the array of diagonal halo points for the neighboring
    /// ComputationalNode which were previously loaded from the GPU memory.
    void* sndDiagHaloPtr = model->getCPUDiagHaloPtr(snd_border);
    /** Obtain a pointer to the temporary array of diagonal halo points for the current
     * ComputationalNode. After data transferring they will replace diagonal halo points
     * which are stored in the GPU memory.
     */
    void* rcvDiagHaloPtr = model->getTmpCPUDiagHaloPtr(rcv_border);
    /// Sending a diagonal halo element to the snd_id ComputationalNode and at the same time,
    /// receiving a diagonal halo element from the rcv_id ComputationalNode.
    mpi_err_status = MPI_Sendrecv(sndDiagHaloPtr, AmountCellsToTransfer, model->MPI_CellType,
            snd_id, globalMPI_id, rcvDiagHaloPtr, AmountCellsToTransfer, model->MPI_CellType,
            rcv_id, rcv_id, MPI_COMM_WORLD, &status);
    /// Check if the MPI transfer was successful
    if(mpi_err_status != MPI_SUCCESS) {
        MPI_Error_string(mpi_err_status, err_buffer, &resultlen);
        throw std::runtime_error(err_buffer);
    }
    /// After receiving the message, check the status to determine
    /// how many cells were actually received
    int number_amount;
    MPI_Get_count(&status, model->MPI_CellType, &number_amount);
    if(number_amount != AmountCellsToTransfer) {
        Log << _WARNING_ << ("Received " + std::to_string(number_amount) +
                " amount of diagonal field cells instead of " + std::to_string(AmountCellsToTransfer));
    }
    Log << "Data has been successfully sent and received.";
    /// Make sure that every ComputationalNode sent and received halo elements
    mpi_err_status = MPI_Barrier(MPI_COMM_COMPUTATIONAL);
    /// Check if the MPI barrier synchronization was successful
    if(mpi_err_status != MPI_SUCCESS) {
        MPI_Error_string(mpi_err_status, err_buffer, &resultlen);
        throw std::runtime_error(err_buffer);
    }
    Log << "Barrier synchronization has been successfully performed";
}

void ComputationalNode::setStopMarker()
{
    MPI_Status status;
    int mpi_err_status, resultlen;
    char err_buffer[MPI_MAX_ERROR_STRING];
    /// The first ComputationalNode sets the stop marker
    if(globalMPI_id == 0) {
        model->setStopMarker();
        Log << "Stop Marker has been successfully set";
    }
    /// Wait for the first ComputationalNode to set the stop marker
    mpi_err_status = MPI_Barrier(MPI_COMM_COMPUTATIONAL);
    /// Check if the MPI barrier synchronization was successful
    if(mpi_err_status != MPI_SUCCESS) {
        MPI_Error_string(mpi_err_status, err_buffer, &resultlen);
        throw std::runtime_error(err_buffer);
    }
    Log << "Barrier synchronization (for the Stop Marker) has been successfully performed";
}
