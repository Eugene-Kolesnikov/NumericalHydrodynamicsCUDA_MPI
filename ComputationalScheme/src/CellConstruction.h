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
#define INSERT_OPER_PLUS_RECURSIVE_1_END
#define INSERT_OPER_PLUS_RECURSIVE_2_END
#define INSERT_OPER_PLUS_RECURSIVE_2(x) lhs.cell.x += rhs.cell.x; INSERT_OPER_PLUS_RECURSIVE_1
#define INSERT_OPER_PLUS_RECURSIVE_1(x) lhs.cell.x += rhs.cell.x; INSERT_OPER_PLUS_RECURSIVE_2
#define INSERT_OPER_MINUS_RECURSIVE_1_END
#define INSERT_OPER_MINUS_RECURSIVE_2_END
#define INSERT_OPER_MINUS_RECURSIVE_2(x) lhs.cell.x -= rhs.cell.x; INSERT_OPER_MINUS_RECURSIVE_1
#define INSERT_OPER_MINUS_RECURSIVE_1(x) lhs.cell.x -= rhs.cell.x; INSERT_OPER_MINUS_RECURSIVE_2
#define INSERT_OPER_MULT_RECURSIVE_1_END
#define INSERT_OPER_MULT_RECURSIVE_2_END
#define INSERT_OPER_MULT_RECURSIVE_2(x) lhs.cell.x *= val; INSERT_OPER_MULT_RECURSIVE_1
#define INSERT_OPER_MULT_RECURSIVE_1(x) lhs.cell.x *= val; INSERT_OPER_MULT_RECURSIVE_2
#define INSERT_OPER_DIV_RECURSIVE_1_END
#define INSERT_OPER_DIV_RECURSIVE_2_END
#define INSERT_OPER_DIV_RECURSIVE_2(x) lhs.cell.x /= val; INSERT_OPER_DIV_RECURSIVE_1
#define INSERT_OPER_DIV_RECURSIVE_1(x) lhs.cell.x /= val; INSERT_OPER_DIV_RECURSIVE_2
#define CREATE_CELL_STRUCT(seq)                                                 \
    struct Cell { CAT(INSERT_MEMBER_RECURSIVE_1 seq, _END) };                   \
    struct cu_Cell { Cell cell;                                                 \
        __device__ cu_Cell(){}                                                  \
        __device__ cu_Cell(Cell c): cell(c) {}                                  \
        friend __device__ cu_Cell operator+(cu_Cell lhs, const cu_Cell& rhs) {  \
            CAT(INSERT_OPER_PLUS_RECURSIVE_1 seq, _END)                         \
            return lhs;                                                         \
        }                                                                       \
        friend __device__ cu_Cell operator-(cu_Cell lhs, const cu_Cell& rhs) {  \
            CAT(INSERT_OPER_MINUS_RECURSIVE_1 seq, _END)                        \
            return lhs;                                                         \
        }                                                                       \
        friend __device__ cu_Cell operator*(cu_Cell lhs, float val) {           \
            CAT(INSERT_OPER_MULT_RECURSIVE_1 seq, _END)                         \
            return lhs;                                                         \
        }                                                                       \
        friend __device__ cu_Cell operator/(cu_Cell lhs, float val) {           \
            CAT(INSERT_OPER_DIV_RECURSIVE_1 seq, _END)                          \
            return lhs;                                                         \
        }                                                                       \
    }


#endif /* CELLCONSTRACTION_H */
