/* 
 * File:   draw_triangle.cpp
 * Author: eugene
 *
 * Created on October 9, 2017, 12:14 AM
 */

#include "interface.h"
#include <cstdio>
#include <cstdlib>
#include "cell.h"

#include <chrono>
#include <thread>

#include <dlfcn.h>

uint64_t p = 2147483647; // = 31, 71;
uint64_t c = 16807; // = 3, 11;

double mapToInterval(uint64_t val, uint64_t max)
{  // transformation to the [0, 1) interval
    return (double)val / (double)max;
}

// Congruential RNG
uint64_t rand_congr()
{
    static unsigned int rnd = 57;  // seed
    rnd = (c * rnd) % p; // generation of a new pseudo-random number
    return rnd;
}

int main()
{
    size_t N_X = 10;
    size_t N_Y = 10;
    Cell* field = new Cell[N_X*N_Y];

    void* m_guiLibHandle = dlopen("libDynamicLibraryVisualization.dylib", RTLD_LOCAL | RTLD_LAZY);
    if (!m_guiLibHandle) {
        fputs (dlerror(), stderr);
        return 1;
    } else {
        printf("Opened the dynamic library.\n");
    }

    bool (*DLV_init)(size_t N_X, size_t N_Y, enum OUTPUT_OPTION outOption) = (bool (*)(size_t N_X, 
            size_t N_Y, enum OUTPUT_OPTION outOption))dlsym(m_guiLibHandle, "DLV_init");
    bool (*DLV_visualize)(void* field, size_t N_X, size_t N_Y) = (bool (*)(void* field,
        size_t N_X, size_t N_Y))dlsym(m_guiLibHandle, "DLV_visualize");
    bool (*DLV_terminate)() = (bool (*)())dlsym(m_guiLibHandle, "DLV_terminate");

    if(!DLV_init(N_X, N_Y, MPEG)) {
        printf("DLV was not able to initialize successfully!\n");
        return 1;
    }

    int frames = 0;
    while(frames++ < 120)
    {
        for(int j = 0; j < N_Y; ++j) {
            for(int i = 0; i < N_X; ++i) {
                field[j*N_X+i].r = mapToInterval(rand_congr(), p);
            }
        }
        
        DLV_visualize((void*)field, N_X, N_Y);
    }

    if(!DLV_terminate()) {
        printf("DLV was not able to terminate successfully!\n");
        return 1;
    }

    dlclose(m_guiLibHandle);

    return 0;
}

