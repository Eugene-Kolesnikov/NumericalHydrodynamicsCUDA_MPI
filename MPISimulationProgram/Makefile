CXX = clang++

LIBS_PATH = -L/usr/local/Cellar/mpich/3.2/lib/

LIBS = -lmpi -lopa -lmpl -lpthread

INCLUDE_PATH = -I/usr/local/Cellar/mpich/3.2/include/ -I/Developer/NVIDIA/CUDA-9.0/include/ -I./../

CXXFLAGS = -c -O3 -std=c++11 -Wempty-body

OBJECTS = \
	src/mpi_main.o \
	src/MPI_Node.o \
	src/ServerNode.o \
	src/ComputationalNode.o \
	../utilities/Logger/src/FileLogger.o \
	../utilities/libLoader/src/libLoader.o \
	../utilities/Register/SystemRegister.o \
	../ComputationalModel/src/ComputationalModel.o \
	../Visualization/src/Visualizer.o


all: $(OBJECTS)
	$(CXX) $(LIBS_PATH) $(LIBS) -o ./build/MPISimulationProgram $(addprefix ./build/,$(notdir $(OBJECTS)))

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $(addprefix ./build/,$(notdir $@))

clean:
	rm ./build/*.o
