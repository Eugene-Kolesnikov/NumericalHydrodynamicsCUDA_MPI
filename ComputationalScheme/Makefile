CXX = clang++

NVCC = nvcc

LIBS = -L/Developer/NVIDIA/CUDA-9.0/lib/ -lcudart -Xlinker -rpath,/Developer/NVIDIA/CUDA-9.0/lib/

CXXFLAGS = -c -O3 -std=c++11

NVCCFLAGS = -c -arch=sm_30 -O3 -ccbin=$(CXX) -std=c++11 --compiler-options '-fPIC'

INCLUDE_PATH = -I/Developer/NVIDIA/CUDA-9.0/include/ -I./../

OBJECTS = \
	src/interface.o \
	src/ComputationalScheme.o \
	src/LBM/LatticeBoltzmannScheme.o \
	src/LBM/LatticeBoltzmannScheme_cpu.o \
	src/UnitTest/UnitTest_MovingBall.o

all: $(OBJECTS)
	$(NVCC) --shared --compiler-options '-fPIC' -o ./build/libComputationalScheme.1.0.0.so $(addprefix ./build/,$(notdir $(OBJECTS))) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) -fpic $< -o $(addprefix ./build/,$(notdir $@))

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_PATH) $< -o $(addprefix ./build/,$(notdir $@))

clean:
	rm ./build/*.o
