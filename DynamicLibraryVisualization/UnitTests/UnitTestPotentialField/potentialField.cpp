/* 
 * File:   potentialField.cpp
 * Author: eugene
 *
 * Created on October 9, 2017, 8:25 AM
 */

#include "interface.h"
#include <cstdio>
#include <cstdlib>
#include "cell.h"
#include <cmath>

#include <dlfcn.h>

size_t xsize = 20000000; // horizontal model size, m
size_t ysize = 20000000;
float dx; // horizontal grid step, m
float dy; // vertical grid step
float* R; // right side of the equation

void initializeField(Cell* field, size_t N_X, size_t N_Y)
{
    dx = xsize / (N_X-1); // horizontal grid step, m
    dy = ysize / (N_Y-1); // vertical grid step
    
    float* x = new float[N_X]; // horizontal coordinates of grid points, m
    float* y = new float[N_Y]; // vertical coordinates of grid points, m
    
    // initialize x,y
    for(int i = 0; i < N_X; ++i) {
        x[i] = i*dx;
    }
    for(int j = 0; j < N_Y; ++j) {
        y[j] = j*dy;
    }
    
    R = new float[N_X*N_Y];
    float* RHO = new float[N_X*N_Y]; // Density (temporary matrix)
    double G = 6.672e-11;
    // compose density array
    float rp = 6000000; // planetary radius, m
    // radius for BC (FI=0)
    float minsize = (xsize > ysize ? ysize : xsize);
    float mindr = (dx > dy ? dy : dx);
    float rBC = minsize / 2.0f - mindr / 2.0f;
    
    for(int j = 0; j < N_Y; ++j) {
        for(int i = 0; i < N_X; ++i) {
            float offset_x = x[i] - xsize / 2.0f;
            float offset_y = y[j] - ysize / 2.0f;
            float rcurrent  = sqrt((offset_x*offset_x + offset_y*offset_y)); // distance from the center of the model
            if(rcurrent > rp) {
                RHO[j*N_X+i] = 0; // outside of the planet
            } else {
                RHO[j*N_X+i] = 5500; // inside the planet
            }
            // initialization of R (result vector)
            if(rcurrent > rBC) { // boundary condition
                R[j*N_X+i] = 0;
            } else {
                R[j*N_X+i] = 2.0f / 3.0f * 4.0f * M_PI * G * RHO[j*N_X+i];
            }
        }
    }
    
    // first guess
    for(int i = 0; i < N_X * N_Y; ++i) {
        field[i].r = 0.0f;
    }
    
    delete[] RHO;
    delete[] x;
    delete[] y;
}

void GaussSeidelSolver(Cell* field, size_t N_X, size_t N_Y)
{
    size_t Iter = 1000; 
    float theta = 1.5f; // relaxation parameter
    for(int I = 0; I < Iter; ++I)
    {
        for(int j = 0; j < N_Y; ++j) 
        {
            for(int i = 0; i < N_X; ++i) 
            {
                if(i == 0 || j == 0 || i == (N_X-1) || j == (N_Y-1)) {
                    field[j*N_X+i].r = 0.0f;
                } else {
                    field[j*N_X+i].r = field[j*N_X+i].r + theta * 
                        ((field[j*N_X+i+1].r - 2*field[j*N_X+i].r + field[j*N_X+i-1].r) / (dx*dx) 
                        + (field[(j+1)*N_X+i].r - 2*field[j*N_X+i].r + field[(j-1)*N_X+i].r) / (dy*dy)
                        - R[j*N_X+i]) / (2.0f/(dx*dx) + 2.0f/(dy*dy));
                }
            }
        }
    }
}

void normalizeField(Cell* field, size_t N_X, size_t N_Y)
{
    float minval = 100000000;
    float maxval = -100000000;
    
    for(int i = 0; i < N_X * N_Y; ++i) {
        if(field[i].r < minval) {
            minval = field[i].r;
        }
    }
    for(int i = 0; i < N_X * N_Y; ++i) {
        field[i].r = (field[i].r - minval); 
    }
    for(int i = 0; i < N_X * N_Y; ++i) {
        if(field[i].r > maxval) {
            maxval = field[i].r;
        }
    }
    for(int i = 0; i < N_X * N_Y; ++i) {
        field[i].r = field[i].r / maxval;
    }   
}

int main()
{
    size_t N_X = 103; // TODO: figure out the problem with the solver when using N_X != N_Y
    size_t N_Y = 110;
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

    if(!DLV_init(N_X, N_Y, PPM)) {
        printf("DLV was not able to initialize successfully!\n");
        return 1;
    }
    
    initializeField(field, N_X, N_Y);
    GaussSeidelSolver(field, N_X, N_Y);
    normalizeField(field, N_X, N_Y);
    
    // rendering of the field and saving
    DLV_visualize((void*)field, N_X, N_Y);

    if(!DLV_terminate()) {
        printf("DLV was not able to terminate successfully!\n");
        return 1;
    }

    dlclose(m_guiLibHandle);
    
    delete[] R;
    delete[] field;

    return 0;
}

