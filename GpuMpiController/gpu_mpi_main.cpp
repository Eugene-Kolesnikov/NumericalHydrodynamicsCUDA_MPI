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
        node = new ServerNode(globalRank, totalNodes, appPath);
    } else {
        // all other nodes are computational nodes
        node = new ComputationalNode(globalRank, totalNodes, appPath);
    }
    
    try {
        /** Initialize the process's environment: 
         * 1) connect all libraries: visualization library, configuration parser, computational library;
         * 2) initialize all necessary variables by parsing the configuration file.
         */
        node->initEnvironment();

        /// Start the application
        node->runNode();
    } catch(...) {
        MPI_Abort();
    }
    
    delete node;

    MPI_Finalize();
    return 0;
}

/*void createMpiStructType()
{
    const int nitems = 4;
    int blocklengths[4] = {1,1,1,1};
    MPI_Datatype types[4] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint offsets[4];

    offsets[0] = offsetof(Cell, r);
    offsets[1] = offsetof(Cell, u);
    offsets[2] = offsetof(Cell, v);
    offsets[3] = offsetof(Cell, e);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_CellType);
    MPI_Type_commit(&MPI_CellType);
}

void parseCmdArgv(int argc, char** argv, int* Nx, int* Ny, std::string& gui_dl)
{
	int c;
    int digit_optind = 0;
	std::string nx("nx");
	std::string ny("ny");
	std::string gui("gui");

	while (1) {
    	int this_option_optind = optind ? optind : 1;
        int option_index = 0;
        static struct option long_options[] = {
            {"gui", required_argument, 0,  0 },
            {"nx",  required_argument, 0,  0 },
            {"ny",  required_argument, 0,  0 },
            {0,         0,             0,  0 }
        };

       c = getopt_long_only(argc, argv, "abc:d:012", long_options, &option_index);

	   if (c == -1)
	   		break;

	   	std::string opt = long_options[option_index].name;
       	switch (c) {
		    case 0:
				if(opt == nx)
					*Nx = atoi(optarg);
				else if(opt == ny)
					*Ny = atoi(optarg);
				else if(opt == gui)
					gui_dl = std::string(optarg);
			default:
            	break;
        }
    }
}*/


