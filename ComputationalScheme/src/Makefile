CXX = g++

CXXFLAGS = -c -O3 -std=c++11

OBJECTS = computationalScheme_interface.o \
	TestScheme.o

all: $(OBJECTS)
	$(CXX) -shared -fpic -o ../build/libComputationalScheme.1.0.0.dylib $(addprefix ../build/,$(OBJECTS))

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -fpic $< -o ../build/$@

clean:
	rm ../build/*.o