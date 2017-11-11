CXX = clang++

LIBS_PATH = -L/usr/local/Cellar/mpich/3.2/lib/

LIBS = -lmpi -lopa -lmpl -lpthread

INCLUDE_PATH = -I/usr/local/Cellar/mpich/3.2/include/

CXXFLAGS = -c -O3 -std=c++11

OBJECTS = gpu_mpi_main.o \
	MPI_Node.o \
	ServerNode.o \
	ComputationalNode.o \
	LoggingSystem/FileLogger.o \
	../../ComputationalModel/src/ComputationalModel.o
	

all: $(OBJECTS)
	$(CXX) $(LIBS_PATH) $(LIBS) -o ../build/simulation_app $(addprefix ../build/,$(notdir $(OBJECTS)))

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $< -o $(addprefix ../build/,$(notdir $@))

clean:
	rm ../build/*.o
