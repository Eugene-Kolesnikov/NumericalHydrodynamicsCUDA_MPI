/*
 * File:   cell.h
 * Author: eugene
 *
 * Created on October 8, 2017, 11:53 PM
 */

#ifndef CELL_H
#define CELL_H

#include "CellConstruction.h"

/*struct Cell {
    double r; // density
    double u; // x-velocity
    double v; // y-velocity
    double e; // energy
};*/


#define STRUCT_DATA_TYPE double
CREATE_CELL_STRUCT((r)(u)(v)(e));


#endif /* CELL_H */
