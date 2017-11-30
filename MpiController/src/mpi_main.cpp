/*
 * File:   gpu_mpi_main.cpp
 * Author: eugene
 *
 * Created on November 1, 2017, 11:17 AM
 */

#include <mpi.h>
#include "ComputationalNode.hpp"
#include "ServerNode.hpp"
#include "MPI_Node.hpp"
#include <string>
#include <exception>

#include <iostream>

int main(int argc, char** argv)
{
    if(argc != 2)
        throw std::runtime_error("Wrong application call (wrong number of arguments)!");
    std::string appPath(argv[1]);

    int globalRank, totalNodes;

    /** Initialize the MPI environment and get the global id of the process
     * and the total amount of processes.
     */
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &globalRank); // id of the current mpi process
    MPI_Comm_size (MPI_COMM_WORLD, &totalNodes); // total amount of processes
    MPI_Errhandler_set(MPI_COMM_WORLD,MPI_ERRORS_RETURN); // return info about errors

    /// Create an empty node class
    MPI_Node* node = nullptr;

    /** Choose what type of node class needs to be created
     * in this particular process id
     */
    if(globalRank == (totalNodes - 1)) {
        // the last node is a server node
        node = new ServerNode(globalRank, totalNodes, appPath, &argc, argv);
    } else {
        // all other nodes are computational nodes
        node = new ComputationalNode(globalRank, totalNodes, appPath, &argc, argv);
    }

    try {
        /** Initialize the process's environment:
         * 1) connect all libraries: visualization library, configuration parser, computational library;
         * 2) initialize all necessary variables by parsing the configuration file.
         */
        node->initEnvironment();

        /// Start the application
        node->runNode();
    } catch(const std::exception& err) {
        std::cout << err.what() << std::endl;
        delete node;
        MPI_Finalize();
        throw;
    }

    delete node;

    MPI_Finalize();
    return 0;
}
