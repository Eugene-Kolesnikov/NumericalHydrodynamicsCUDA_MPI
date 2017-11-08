/* 
 * File:   cell.h
 * Author: eugene
 *
 * Created on October 8, 2017, 11:53 PM
 */

#ifndef CELL_H
#define CELL_H

#define MAX_CELL_ARG 10
#define STRUCT_DATA_TYPE double

#define PRIMITIVE_CAT(x, y) x ## y
#define CAT(x, y) PRIMITIVE_CAT(x, y)
#define INSERT_MEMBER_RECURSIVE_1_END
#define INSERT_MEMBER_RECURSIVE_2_END
#define INSERT_MEMBER_RECURSIVE_2(x) STRUCT_DATA_TYPE x; INSERT_MEMBER_RECURSIVE_1
#define INSERT_MEMBER_RECURSIVE_1(x) STRUCT_DATA_TYPE x; INSERT_MEMBER_RECURSIVE_2
#define CREATE_CELL_STRUCT(name,seq) struct name { CAT(INSERT_MEMBER_RECURSIVE_1 seq, _END) }

#ifdef __cplusplus
extern "C" {
#endif

/*struct Cell {
    double r; // density
    double u; // x-velocity
    double v; // y-velocity
    double e; // energy
};*/

CREATE_CELL_STRUCT(Cell,(r)(u)(v)(e));


#ifdef __cplusplus
}
#endif

#endif /* CELL_H */

