/**
* @file interface.h
* @brief This header file contains the interface to the Visualization module.
 * It consists of three functions: initialization, visualization, termination.
* @author Eugene Kolesnikov 
* @date 8/10/2017 
*/

#ifndef INTERFACE_H
#define INTERFACE_H

#include <stdlib.h>
#include "offscreen.h"

#ifdef __cplusplus
extern "C" {
#endif

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
 * @return a bool parameter which indicates if the function performed successfully (true) or not (false). 
 */
bool DLV_init(size_t N_X, size_t N_Y, enum OUTPUT_OPTION outOption);

/** 
 * @brief The server node calls the `DLV_visualize` function each time it is necessary to
 * plot hydrodynamics fields: density, velocity,... (depends on the computational model).
 * The decision of what property of the field must be rendered is specified in the DLV library.
 * @param field -- pointer to the field.
 * @param N_X -- discretization of the grid along the X direction.
 * @param N_Y -- discretization of the grid along the Y direction.
 * @return a bool parameter which indicates if the function performed successfully (true) or not (false). 
*/
bool DLV_visualize(void* field, size_t N_X, size_t N_Y);

/** 
 * @brief The server node calls the `DLV_terminate` function once when the computations
 * has already finished to deinitialize the environment.
 * @return a bool parameter which indicates if the function performed successfully (true) or not (false). 
*/
 bool DLV_terminate();


#ifdef __cplusplus
}
#endif

#endif /* INTERFACE_H */

