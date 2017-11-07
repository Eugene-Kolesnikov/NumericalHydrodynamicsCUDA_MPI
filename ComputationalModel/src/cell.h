/* 
 * File:   cell.h
 * Author: eugene
 *
 * Created on October 8, 2017, 11:53 PM
 */

#ifndef CELL_H
#define CELL_H

#ifdef __cplusplus
extern "C" {
#endif

struct Cell {
    double r; // density
    double u; // x-velocity
    double v; // y-velocity
    double e; // energy
};


#ifdef __cplusplus
}
#endif

#endif /* CELL_H */

