/**
* @file interface.cpp
* @brief This file contains description of functions of the Visualization module.
 * It consists of three functions: initialization, visualization, termination.
* @author Eugene Kolesnikov 
* @date 8/10/2017 
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

/** 
 * @brief The server node calls the `DLV_init` function once before the computations
 * start in order to initialize the environment: create the window where OpenGL
 * will render the field, allocate necessary memory, initialize environmental variables,
 * set the output method (PPM, PNG, MPEG).
 * @param N_X -- discretization of the grid along the X direction.
 * @param N_Y -- discretization of the grid along the Y direction.
 * @param outOption -- parameter specifies what output is expected from the DLV: 
 * 'PPM' creates a .ppm image file, 'PNG' creates a .png image file, 
 * 'MPEG' creates a .mpeg video file. In the last case, the function initializes the 
 * file for further writing frames into it.
 * @return a bool parameter which indicates if the function performed successfully (true) or not (false). */
bool DLV_init(size_t N_X, size_t N_Y, enum OUTPUT_OPTION outOption, const char* path)
{
    /// Create the window with an OpenGL context of the 'MainWindow' class
    mainWindow = new MainWindow;
    mainWindow->create("Visualization");

    /// Set up the background color of the window (Dark blue)
    mainWindow->setBGColor(0.0f, 0.0f, 0.4f, 0.0f);

    /// Allocate memory for the FieldObject
    fieldObject = new FieldObject(path);
    /// Set up the output method
    fieldObject->setOutputOption(outOption);
    /// Initialize the FiledObject
    fieldObject->init(N_X, N_Y);

    return SUCCESS;
}

/** 
 * @brief The server node calls the `DLV_visualize` function each time it is necessary to
 * plot hydrodynamics fields: density, velocity,... (depends on the computational model).
 * The decision of what property of the field must be rendered is specified in the DLV library.
 * @param field -- pointer to the field.
 * @param N_X -- discretization of the grid along the X direction.
 * @param N_Y -- discretization of the grid along the Y direction.
 * @return a bool parameter which indicates if the function performed successfully (true) or not (false). */
bool DLV_visualize(void* field, size_t N_X, size_t N_Y)
{
    if(glfwWindowShouldClose(mainWindow->window()) == 0) {
        /// If the window shouldn't be closed, clear the screen and render the field
        mainWindow->clearScreen();
        fieldObject->render(field, N_X, N_Y);
        // Swap buffers
        glfwSwapBuffers(mainWindow->window());
        glfwPollEvents();
        return SUCCESS;
    } else {
        /// Return FAIL if the window should be closed but the rendering is required
        return FAIL;
    }
}

/** 
 * @brief The server node calls the `DLV_terminate` function once when the computations
 * has already finished to deinitialize the environment.
 * @return a bool parameter which indicates if the function performed successfully (true) or not (false). */
 bool DLV_terminate()
 {
     try {
         /// Delete the FieldObject, close the MainWindow
        delete fieldObject;
        mainWindow->terminate();
        delete mainWindow;
        return SUCCESS;
     } catch(...) {
         /// Return FAIL if any runtime exception caught
         return FAIL;
     }
 }
