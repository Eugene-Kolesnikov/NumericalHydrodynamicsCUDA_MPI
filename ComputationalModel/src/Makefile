CXX = g++

LIBS_PATH = -L/usr/local/Cellar/mpich/3.2/lib/

LIBS = -lmpi -lopa -lmpl -lpthread

INCLUDE_PATH = -I/usr/local/Cellar/mpich/3.2/include/

CXXFLAGS = -c -O3 -std=c++11

OBJECTS = computationalModel_interface.o \
	CPUTestComputationalModel.o \
	../../MpiController/src/LoggingSystem/FileLogger.o

all: $(OBJECTS)
	$(CXX) -shared -fpic -o ../build/libComputationalModel.1.0.0.dylib $(addprefix ../build/,$(OBJECTS)) $(LIBS_PATH) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) -fpic $< -o ../build/$@

clean:
	rm ../build/*.o