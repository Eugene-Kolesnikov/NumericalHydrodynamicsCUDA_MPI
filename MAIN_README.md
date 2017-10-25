# Numerical Hydrodynamics MPI+CUDA Project
## Description of the simulation software

The idea of the software is its modularity, which in this particular case means that it consists of 3 absolutely independent sections: 

- **Main System (MS)**, which is responsible for all data transportation among nodes, logic of the program, CPU-GPU-MPI interconnection and logging system;
- **Dynamic Library of Visualization (DLV)**. It has a pre-defined interface (MS calls a particular function with pre-defined arguments each time the visualization of a frame is needed). It is up to the DLV to decide how exactly to represent the data;
- **Dynamic Library of Initialization and Computational Kernel (DLICK)**. Two problem-specific functions: initialization and scheme are also are separated to another module.
- **Dynamic library of models**. The module which contains different cell structures and necessary back-end functions which are required for different models. 
- **Configuration file**. CPU-GPU-MPI parameters, as well as algorithm and scheme specific parameters

![MainDiagram](../../images/MainDiagram.png)

#### Main System (MS)
The main section of the program is the only one that the user can not change. If DLV and DLICK are just dynamic libraries with predefined interface, they therefore can be changed with another libraries which have the same interface, and the program will perfectly work, whereas MS is the executable part of the code which can't be changed by a user.

Because the software is developed using CPU-GPU-MPI interaction, it uses a lot of different parameters, which can and must be chosen specifically for each problem. Therefore, there is a special configuration file, which contains all necessary variables which can be modified by a user.

Initialization of the MS is done each time the program launches by parsing the configure file, so there is no need to recompile the program in order to update parameters.


#### Dynamic Library of Visualization (DLV)
In this particular case, DLV uses OpenGL to render the field and saves it to one of the following files: ppm, png, mpeg (for videos). Since the software solves 2D problems, the Visualization is a 2D image (static or dynamic depending on the problem), which is generated using OpenGL 4.x and GLFW 3 as an interface generation library.

Interface functions:

```{python}
bool DLV_init(size_t N_X, size_t N_Y, enum OUTPUT_OPTION outOption);
bool DLV_visualize(void* field, size_t N_X, size_t N_Y);
bool DLV_terminate();
```

The field should be normalized to [0,1] before sending it to the ```DLV_visualize``` function. The DLV renders the field on a square \f$[0,1]^2\f$.
\f[ DLV: [0,1] \rightarrow [0,1]^3 \f]

#### Dynamic Library of Initialization and Computational Kernel (DLICK)
This is the main computational module, since different models have a lot of different schemes (ways of using the model to solve a particular problem, as well as boundary conditions). Moreover, this module also contains manually defined initialization functions.

#### Dynamic Library of Models (DLM)
Different algorithms require different models of grid nodes, therefore several of this structures are implemented in order for the user to choose the one for his problem.

Unfortunately, it seems impossible to separate models from the MS, because different schemes and algorithms require different set of parameters in grid cells, which at the same time requires to use different structs in the code. Exchange of this structs in real time seems way too much difficult to implement, since it requires to work with virtual classes, inheritance, and polymorphism. Even though these mechanisms are very easy on their own, it is quite challenging to understand how they will be transfered among cpu nodes and gpu devices. One can be said for sure, using polymorphism is much more computationally difficult than just using C-style structures.

#### Configuration file

```
TAU = 1.0e-5 % time step
TOTAL_TIME = 5.0e+0 % total time from 0 to TOTAL_TIME with the step TAU
STEP_LENGTH = 100 % loop steps skipping before each visualization
N_X -- discretization of the grid along the X direction
N_Y -- discretization of the grid along the Y direction
CUDA_X_THREADS -- number of threads in a CUDA block along the X direction
CUDA_Y_THREADS -- number of threads in a CUDA block along the Y direction
MPI_NODES_X -- MPI nodes along the X direciton
MPI_NODES_Y -- MPI nodes along the Y direciton
...
```
In order to create ```MPI_NODES_X * MPI_NODES_Y``` MPI nodes, the program calls another program using ```sys()``` function. Maybe it's better to create a GUI for the configuration file.


## Computational sequence

1. Pre-processing: 
2. Processing: 
3. Post-processing: 


#TODO:
- Learn how to use Doxygen