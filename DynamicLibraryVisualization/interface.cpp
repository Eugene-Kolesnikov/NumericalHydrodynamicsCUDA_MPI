/* 
 * File:   MainWindow.cpp
 * Author: eugene
 * 
 * Created on October 8, 2017, 11:50 PM
 */

#include "interface.h"
#include <GLFW/glfw3.h>
#include <exception>
#include "MainWindow.hpp"
#include "FieldObject.hpp"

#define SUCCESS true
#define FAIL false

MainWindow* mainWindow;
FieldObject* fieldObject;

/* The server node calls the `DLV_init` function once before the computations...
 * start in order to initialize the environment.
 * The function returns one bool parameter: the function performed successfully or not. */
bool DLV_init(size_t N_X, size_t N_Y, enum OUTPUT_OPTION outOption)
{
    try {
        mainWindow = new MainWindow;
    	mainWindow->create("Simple Rendering Scene");

    	// Dark blue background
    	mainWindow->setBGColor(0.0f, 0.0f, 0.4f, 0.0f);

    	fieldObject = new FieldObject;
        fieldObject->setOutputOption(outOption);
    	fieldObject->init(N_X, N_Y);
    } catch(std::runtime_error err) {
        printf("Error: (DLV_init) %s\n", err.what());
        return FAIL;
    }

    return SUCCESS;
}

/* The server node calls the `DLV_visualize` function each time it is necessary to...
 * plot hydrodynamics fields: density, velocity,... (depends on the computational model).
 * The function takes parameters: pointer to the field [void* field],...
 * discretization of the grid along the X direction [size_t N_X],...
 * discretization of the grid along the Y direction [size_t N_Y].
 * The function returns one bool parameter: the function performed successfully or not. */
bool DLV_visualize(void* field, size_t N_X, size_t N_Y)
{
    if(glfwWindowShouldClose(mainWindow->window()) == 0) {
        mainWindow->clearScreen();
        fieldObject->render(field, N_X, N_Y);
        // Swap buffers
        glfwSwapBuffers(mainWindow->window());
        glfwPollEvents();
        return SUCCESS;
    } else {
        return FAIL;
    }
}

/* The server node calls the `DLV_visualize` function once when the computations...
 * finish to deinitialize the environment.
 * The function returns one bool parameter: the function performed successfully or not. */
 bool DLV_terminate()
 {
     delete fieldObject;
     mainWindow->terminate();
     delete mainWindow;

     return SUCCESS;
 }
