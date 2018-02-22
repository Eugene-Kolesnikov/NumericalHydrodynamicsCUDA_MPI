##Tasks:
1. [x] `Change projects' names`.
* [x] Put the compiled files in a separate folder and change extentions of dynamic libraries to `.so`.
* [x] Rearrange source code tree by adding `include` folders and add links to makefiles to be able to call `#include <ComputationalModel/include/computationalModel_interface.h>`.
* [x] Add `extern "C"` to interfaces.
* [ ] Use `libLoader` to load dynamic libs (Computational Model)
* [ ] Add a separate xml file `SystemConfig` with names of dynamic libraries.
* [ ] Add `Register`
* [ ] Create different git branches for 'LB development' and 'MIC (Eulerian-Lagrangian) model development'.
* [ ] `Program modifications`.
* [x] Understand how to link C++ classes
	* http://www.linuxjournal.com/article/3687
* [ ] Create separate 'no MPI' program. (Run computations in a separate thread).
* [ ] `Core vizualization`.
* [ ] `Vizualizer` and `Data Vizualizer`.
* [ ] Update Config parser.
* [ ] `Scene Generator`.
* [ ] `Mesh Generator`.
* [ ] Add an option of simultaneously running computations on GPU and CPU with OpenMP.
* [ ] `Lattice Boltzmann Method`.
* [ ] Quad-tree lattice.
* [ ] Create a better system for debugging and unit testing.
* [ ] Finite Element Method.
* [ ] Benchmark to understand the optimal configuration parameters (MPI Nodes, GPU Threads, CPU Threads) for the particular system.
* [ ] Compiler helper (GUI for generation of make file and compilation).

##Change projects' names:

* [x] `MpiController` $\rightarrow$ `MPISimulationProgram`
* [x] `StartInterface` $\rightarrow$ `x` (NumericalSolver2D)

##Program modifications:
* [ ] Add an option to choose types of the program: `single server`, `all servers (each node is a server and saves data to its own file)`.
* [ ] Create a `enum HaloType = {CU_LEFT_BORDER = 0, CU_RIGHT_BORDER = 1, CU_TOP_BORDER = 2, CU_BOTTOM_BORDER = 3, CU_LEFT_TOP_BORDER = 0, CU_RIGHT_TOP_BORDER = 1, CU_LEFT_BOTTOM_BORDER = 2, CU_RIGHT_BOTTOM_BORDER = 3}`. If it's impossible to make to enum elements have the same value, make this: `enum HaloType = {CU_LEFT_TOP_BORDER = 0, CU_RIGHT_TOP_BORDER = 1, CU_LEFT_BOTTOM_BORDER = 2, CU_RIGHT_BOTTOM_BORDER = 3, CU_LEFT_BORDER = 4, CU_RIGHT_BORDER = 5, CU_TOP_BORDER = 6, CU_BOTTOM_BORDER = 7}`.
* [ ] Add global static register for global variables: `N_X, N_Y, mpi_x, mpi_y, X_MAX, Y_MAX, TAU,...`.
* [ ]  For the server node: `QApplication` at the beginning of the program, run logic in a separate thread and return `app.exec()`.
* [ ]  Update CellConstructor to create Cell struct with different field types.
* [ ]  Create separate class `ComputationalModelController` between `MPINode` and `ComputationalModel` which regulates the job between the `CPUComputationalModel` and `GPUComputationalModel`.
* `MPINode`:
	* [ ] Add MPI calls wrappers `MPIController` for a simpler understanding of the code.
* `ComputationalNode`: 
	* [ ] 	Add `enum ComputationalType = {Eulerian, Lagrangian, EulerianLagrangian}`.
	* [ ]  Add the step of exchanging values of the amounts of cells which will be transfered between nodes (Lagrangian).
* `ComputationalScheme`:
	* [ ] Create a function `allocateField(enum memoryType = {CPU, CPU_PageLocked, GPU})`.
	* [ ] Create a function `allocateArray(enum memoryType = {CPU, CPU_PageLocked, GPU})` for halos.
	* [ ] Change functions `performCPUSimulationStep`, `performGPUSimulationStep`, `updateCPUGlobalBorders`, `updateGPUGlobalBorders` to respective functions `performSimulationStep(enum SimulationType = {CPU, GPU})`, `updateGlobalBorders(enum SimulationType = {CPU, GPU})`.
* `Computational Model`:
	* [ ] Replace `void* [ ] getTmpCPUHaloPtr(size_t border_type)`, `void* [ ] getTmpCPUDiagHaloPtr(size_t border_type)` with one function `getTmpCPUHaloPtr(enum HaloType {border, diagonal}, size_t border_type)`.
	* [ ] Replace `void* [ ] getCPUHaloPtr(size_t border_type)`, `void* [ ] getCPUDiagHaloPtr(size_t border_type)` with one function `getCPUHaloPtr(enum HaloType {border, diagonal}, size_t border_type)`.
* `GPUComputationalModel`:
	* [ ] Create a faster version of `updateGPUHaloElements_kernel`. Use enough threads to update all elements at once.
	* [ ] Create a `__host__ __device__` wrapper for cu_field,... and overload `operator()(x,y)` and `operator()(i)`.
	* [ ] Add GPU calls wrappers `CUDAController, OpenCLController`.

## StartInerface:
* [ ] Create a name for the program and change the GUI.
* [ ] Option to choose the MPI execution app.
* [ ] Button which opens a window where one can choose the GPU configuration parameters. The program must automatically detect GPUs and understand types of architectures.
* [ ] Button to open a `DataReader` and `Scene Generator`.
* [ ] Initial window asks to give the name for the project and the location of files where result, configuration file, scene file, and mesh file (each file must be reusable by `DataVisualizer`, `SceneCreator`, `MeshGenerator`, `StartInterface`) will be saved.

	
## Scene Generator:
* [ ] Resources: 
	* https://doc.qt.io/qt-5.10/qt3d-scene2d-example.html
	* http://doc.qt.io/qt-5/qtopengl-2dpainting-example.html
* [ ] Automatic edge detector and vertex detector.
* [ ] Axes in the right-bottom corner, border of the global area.
* [ ] Button for the Mesh Generator (create as a separate program).
* [ ] Add a ruler for the cursor.

## Mesh Generator:
* [ ] Unified generator for Eulerian Cells and Lagrangian points

## Vizualizer:
* [ ] Output file in the binary form.
* [ ] Compile `QCustomPlot` as a separate dynamic library.

## Data visualizer:
* [ ] Add support for an autoplay (video).
* [ ] Export as a video.
* [ ] Save frame as a png image.
* [ ] Add vector fields.
* [ ] At the beginning, upload only the first frame and dynamically load the frame of interest to minimize the memory requirement.
* [ ] Create a separate window for video output (5 frames in active buffer, 5 frames in another thread in reserve buffer). Main thread plots field from acrive buffer, the other one reads data from the report file to the reserve buffer. Automatically (in separate thread) load 5 frames in reserve buffer after that swap pointers to active and reserve buffers.

##Vizualization Core:
* [ ] Create a general core for `Vizualizer` and `DataVizualizer`.
* [ ] Uniform visualization structure for Eulerian and Lagrangian schemes. Write in the report file `Cell1, Cell2,...` instead of `Cell1.x Cell2.x,...,Cell1.y,Cell2.y,...`.

## Lattice Boltzmann Method:
* [ ] Create a base class which performes boundary conditions
* [ ] LBM with porous media.