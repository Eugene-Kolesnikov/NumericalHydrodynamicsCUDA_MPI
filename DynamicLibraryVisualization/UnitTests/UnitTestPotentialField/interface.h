/* 
 * File:   interface.h
 * Author: eugene
 *
 * Created on October 8, 2017, 11:47 PM
 */

#ifndef INTERFACE_H
#define INTERFACE_H

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif
    
enum OUTPUT_OPTION { PPM, PNG, MPEG };

/* The server node calls the `DLV_init` function once before the computations...
 * start in order to initialize the environment.
 * The function returns one bool parameter: the function performed successfully or not. */
bool DLV_init(size_t N_X, size_t N_Y, enum OUTPUT_OPTION outOption);

/* The server node calls the `DLV_visualize` function each time it is necessary to...
 * plot hydrodynamics fields: density, velocity,... (depends on the computational model).
 * The function takes parameters: pointer to the field [void* field],...
 * discretization of the grid along the X direction [size_t N_X],...
 * discretization of the grid along the Y direction [size_t N_Y].
 * The function returns one bool parameter: the function performed successfully or not. */
bool DLV_visualize(void* field, size_t N_X, size_t N_Y);

/* The server node calls the `DLV_visualize` function once when the computations...
 * finish to deinitialize the environment.
 * The function returns one bool parameter: the function performed successfully or not. */
 bool DLV_terminate();


#ifdef __cplusplus
}
#endif

#endif /* INTERFACE_H */

