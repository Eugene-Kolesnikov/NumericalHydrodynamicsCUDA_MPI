/*
 * File:   CellConstraction.h
 * Author: eugene
 *
 * Created on October 8, 2017, 11:53 PM
 */

#ifndef CELLCONSTRACTION_H
#define CELLCONSTRACTION_H

#define PRIMITIVE_CAT(x, y) x ## y
#define CAT(x, y) PRIMITIVE_CAT(x, y)
#define INSERT_MEMBER_RECURSIVE_1_END
#define INSERT_MEMBER_RECURSIVE_2_END
#define INSERT_MEMBER_RECURSIVE_2(x) STRUCT_DATA_TYPE x; INSERT_MEMBER_RECURSIVE_1
#define INSERT_MEMBER_RECURSIVE_1(x) STRUCT_DATA_TYPE x; INSERT_MEMBER_RECURSIVE_2
#define CREATE_CELL_STRUCT(seq)                                                 \
    struct Cell { CAT(INSERT_MEMBER_RECURSIVE_1 seq, _END) };


#endif /* CELLCONSTRACTION_H */
