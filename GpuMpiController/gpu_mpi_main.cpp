/* 
 * File:   gpu_mpi_main.cpp
 * Author: eugene
 *
 * Created on November 1, 2017, 11:17 AM
 */

#include <mpi.h>
#include "processNode.hpp"
#include "serverNode.hpp"
#include "MPI_Node.hpp"
#include <getopt.h>
#include <stdlib.h>

MPI_Datatype MPI_CellType;

void createMpiStructType();
void parseCmdArgv(int argc, char** argv, int* Nx, int* Ny, std::string& gui_dl);

int rank, size;
int main(int argc, char** argv) 
{
    int Nx = -1, Ny = -1;
    std::string gui_dl;

    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank); // id of the current mpi process
    MPI_Comm_size (MPI_COMM_WORLD, &size); // total amount of processes

    createMpiStructType(); // create necessary `MPI_CellType` for MPI transfer system

    parseCmdArgv(argc, argv, &Nx, &Ny, gui_dl);
    if((Nx == -1 || Ny == -1) && rank == size - 1) {
	fputs("Error: not enough cmd arguments!\n", stderr);
	MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    MPI_Node* 

    if(rank < size - 1) { // computational nodes
	ProcessNode* process = new ProcessNode(rank,size,Nx,Ny);
        process->loadXMLParserLib();
        process->initEnvironment();
	process->runNode();
	delete process;
    } else { // server node
	ServerNode* server = new ServerNode(rank,size,Nx,Ny);
	server->setArgcArgv(argc, argv);
	server->loadVisualizationLib(); // \TODO: open the dynamic lib using the full path
        server->loadXMLParserLib();
        server->initEnvironment();
	server->runNode();
	delete server;
    }

    MPI_Finalize();
    return 0;
}

void createMpiStructType()
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
}


