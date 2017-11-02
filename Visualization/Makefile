CXX = g++
CC = gcc

LIBS =  -lglfw3 -fobjc-arc -framework Cocoa -framework OpenGL -framework IOKit -framework CoreVideo -lpng -L/usr/local/Cellar/ffmpeg/3.0.1/lib -lavcodec -lavutil -lswscale

CXXFLAGS = -c -O3 -std=c++11
CCFLAGS = -c -O3

OBJECTS = MainWindow.o \
	FieldObject.o \
	offscreen.o \
	shader.o \
	interface.o
	

all: $(OBJECTS)
	$(CXX) -shared -fpic -o libVisuzalization.1.0.0.dylib $(OBJECTS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -fpic $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -fpic $< -o $@

clean:
	rm *.o
