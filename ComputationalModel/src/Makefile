CXX = g++

NVCC = nvcc

LIBS_PATH = -L/usr/local/Cellar/mpich/3.2/lib/

LIBS = -lmpi -lopa -lmpl -lpthread -L/Developer/NVIDIA/CUDA-9.0/lib/ -lcudart

INCLUDE_PATH = -I/usr/local/Cellar/mpich/3.2/include/ -I/Developer/NVIDIA/CUDA-9.0/include/

CXXFLAGS = -c -O3 -std=c++11

NVCCFLAGS = -c -arch=sm_30 -O3 -ccbin=$(CXX) -std=c++11 --compiler-options '-fPIC'

OBJECTS = computationalModel_interface.o \
	CPUComputationalModel.o \
	GPUComputationalModel.o \
	ComputationalModel.o \
	../../MpiController/src/FileLogger.o

all: $(OBJECTS)
	$(NVCC) --shared --compiler-options '-fPIC' -o ../build/libComputationalModel.1.0.0.dylib $(addprefix ../build/,$(notdir $(OBJECTS))) $(LIBS_PATH) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) -fpic $< -o $(addprefix ../build/,$(notdir $@))

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_PATH) $< -o $(addprefix ../build/,$(notdir $@))

clean:
	rm ../build/*.o
