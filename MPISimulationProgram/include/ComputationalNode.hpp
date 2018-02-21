/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   ProcessNode.hpp
 * Author: eugene
 *
 * Created on November 1, 2017, 12:49 PM
 */

#ifndef COMPUTATIONALNODE_HPP
#define COMPUTATIONALNODE_HPP

#include <MPISimulationProgram/include/MPI_Node.hpp>
#include <string>
#include <cstdlib>
#include <mpi.h>

class ComputationalNode : public MPI_Node {
public:
    ComputationalNode(size_t globalRank, size_t totalNodes, std::string app_path, int* _argc, char** _argv);
    virtual ~ComputationalNode();

    virtual void initEnvironment();
    virtual void runNode();

protected:
    void createComputationalMPIgroup();
    void loadInitSubFieldFromServer();
    void sendUpdatedSubFieldToServer();
    void shareHaloElements();
    void sndRcvHaloElements(int snd_id, int rvc_id, int snd_border, int rcv_border);
    void sndRcvDiagHaloElements(int snd_id, int rcv_id, int snd_border, int rcv_border);
    void setStopMarker();

protected:
    int* compRanks;
    MPI_Group world_group;
    MPI_Group computational_group;
    MPI_Comm MPI_COMM_COMPUTATIONAL;
};

#endif /* COMPUTATIONALNODE_HPP */
