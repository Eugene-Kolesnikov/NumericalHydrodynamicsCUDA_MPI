/*
 * File:   CellConstraction.h
 * Author: eugene
 *
 * Created on October 8, 2017, 11:53 PM
 */

#ifndef CELLCONSTRACTION_H
#define CELLCONSTRACTION_H

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <typeinfo>
#include <typeindex>
#include <vector>

#define PRIMITIVE_CAT(x, y) x ## y
#define CAT(x, y) PRIMITIVE_CAT(x, y)

#define INSERT_MEMBER_RECURSIVE_1_END
#define INSERT_MEMBER_RECURSIVE_2_END

#define INSERT_1_MEMBERS(x, y) STRUCT_DATA_TYPE x
#define INSERT_2_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_3_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_4_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_5_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_6_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_7_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_8_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_9_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_10_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_11_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_12_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_13_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_14_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_15_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_16_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_17_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_18_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_19_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_20_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_21_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_22_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_23_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_24_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_25_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_26_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_27_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_28_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_29_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_30_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_31_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_32_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_33_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_34_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_35_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_36_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_37_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_38_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_39_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_40_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_41_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_42_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_43_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_44_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_45_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_46_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_47_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_48_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_49_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_50_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_51_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_52_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_53_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_54_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_55_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_56_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_57_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_58_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_59_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_60_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_61_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_62_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]
#define INSERT_63_MEMBERS(x, y) STRUCT_DATA_TYPE x[y]

#define INSERT_MEMBER_RECURSIVE_2(x, y) INSERT_##y##_MEMBERS(x,y); INSERT_MEMBER_RECURSIVE_1
#define INSERT_MEMBER_RECURSIVE_1(x, y) INSERT_##y##_MEMBERS(x,y); INSERT_MEMBER_RECURSIVE_2

#define COUNT_MEMBERS_1_END
#define COUNT_MEMBERS_2_END
#define COUNT_MEMBERS_1(x, y) + 1 COUNT_MEMBERS_2
#define COUNT_MEMBERS_2(x, y) + 1 COUNT_MEMBERS_1
#define COUNT_MEMBERS(x, y) 1 COUNT_MEMBERS_1

#define ARRAY_0_MEMBERS_END
#define ARRAY_1_MEMBERS_END
#define ARRAY_2_MEMBERS_END
#define ARRAY_3_MEMBERS_END
#define ARRAY_4_MEMBERS_END
#define ARRAY_5_MEMBERS_END
#define ARRAY_6_MEMBERS_END
#define ARRAY_7_MEMBERS_END
#define ARRAY_8_MEMBERS_END
#define ARRAY_9_MEMBERS_END
#define ARRAY_10_MEMBERS_END
#define ARRAY_11_MEMBERS_END
#define ARRAY_12_MEMBERS_END
#define ARRAY_13_MEMBERS_END
#define ARRAY_14_MEMBERS_END
#define ARRAY_15_MEMBERS_END
#define ARRAY_16_MEMBERS_END
#define ARRAY_17_MEMBERS_END
#define ARRAY_18_MEMBERS_END
#define ARRAY_19_MEMBERS_END
#define ARRAY_20_MEMBERS_END
#define ARRAY_21_MEMBERS_END
#define ARRAY_22_MEMBERS_END
#define ARRAY_23_MEMBERS_END
#define ARRAY_24_MEMBERS_END
#define ARRAY_25_MEMBERS_END
#define ARRAY_26_MEMBERS_END
#define ARRAY_27_MEMBERS_END
#define ARRAY_28_MEMBERS_END
#define ARRAY_29_MEMBERS_END
#define ARRAY_30_MEMBERS_END
#define ARRAY_31_MEMBERS_END
#define ARRAY_32_MEMBERS_END
#define ARRAY_33_MEMBERS_END
#define ARRAY_34_MEMBERS_END
#define ARRAY_35_MEMBERS_END
#define ARRAY_36_MEMBERS_END
#define ARRAY_37_MEMBERS_END
#define ARRAY_38_MEMBERS_END
#define ARRAY_39_MEMBERS_END
#define ARRAY_40_MEMBERS_END
#define ARRAY_41_MEMBERS_END
#define ARRAY_42_MEMBERS_END
#define ARRAY_43_MEMBERS_END
#define ARRAY_44_MEMBERS_END
#define ARRAY_45_MEMBERS_END
#define ARRAY_46_MEMBERS_END
#define ARRAY_47_MEMBERS_END
#define ARRAY_48_MEMBERS_END
#define ARRAY_49_MEMBERS_END
#define ARRAY_50_MEMBERS_END
#define ARRAY_51_MEMBERS_END
#define ARRAY_52_MEMBERS_END
#define ARRAY_53_MEMBERS_END
#define ARRAY_54_MEMBERS_END
#define ARRAY_55_MEMBERS_END
#define ARRAY_56_MEMBERS_END
#define ARRAY_57_MEMBERS_END
#define ARRAY_58_MEMBERS_END
#define ARRAY_59_MEMBERS_END
#define ARRAY_60_MEMBERS_END
#define ARRAY_61_MEMBERS_END
#define ARRAY_62_MEMBERS_END
#define ARRAY_63_MEMBERS_END
#define ARRAY_64_MEMBERS_END

#define ARRAY_0_MEMBERS(x, y) AmountOfArrayMembers[0] = y; CellOffsets[0] = offsetof(Cell, x); ARRAY_1_MEMBERS
#define ARRAY_1_MEMBERS(x, y) AmountOfArrayMembers[1] = y; CellOffsets[1] = offsetof(Cell, x); ARRAY_2_MEMBERS
#define ARRAY_2_MEMBERS(x, y) AmountOfArrayMembers[2] = y; CellOffsets[2] = offsetof(Cell, x); ARRAY_3_MEMBERS
#define ARRAY_3_MEMBERS(x, y) AmountOfArrayMembers[3] = y; CellOffsets[3] = offsetof(Cell, x); ARRAY_4_MEMBERS
#define ARRAY_4_MEMBERS(x, y) AmountOfArrayMembers[4] = y; CellOffsets[4] = offsetof(Cell, x); ARRAY_5_MEMBERS
#define ARRAY_5_MEMBERS(x, y) AmountOfArrayMembers[5] = y; CellOffsets[5] = offsetof(Cell, x); ARRAY_6_MEMBERS
#define ARRAY_6_MEMBERS(x, y) AmountOfArrayMembers[6] = y; CellOffsets[6] = offsetof(Cell, x); ARRAY_7_MEMBERS
#define ARRAY_7_MEMBERS(x, y) AmountOfArrayMembers[7] = y; CellOffsets[7] = offsetof(Cell, x); ARRAY_8_MEMBERS
#define ARRAY_8_MEMBERS(x, y) AmountOfArrayMembers[8] = y; CellOffsets[8] = offsetof(Cell, x); ARRAY_9_MEMBERS
#define ARRAY_9_MEMBERS(x, y) AmountOfArrayMembers[9] = y; CellOffsets[9] = offsetof(Cell, x); ARRAY_10_MEMBERS
#define ARRAY_10_MEMBERS(x, y) AmountOfArrayMembers[10] = y; CellOffsets[10] = offsetof(Cell, x); ARRAY_11_MEMBERS
#define ARRAY_11_MEMBERS(x, y) AmountOfArrayMembers[11] = y; CellOffsets[11] = offsetof(Cell, x); ARRAY_12_MEMBERS
#define ARRAY_12_MEMBERS(x, y) AmountOfArrayMembers[12] = y; CellOffsets[12] = offsetof(Cell, x); ARRAY_13_MEMBERS
#define ARRAY_13_MEMBERS(x, y) AmountOfArrayMembers[13] = y; CellOffsets[13] = offsetof(Cell, x); ARRAY_14_MEMBERS
#define ARRAY_14_MEMBERS(x, y) AmountOfArrayMembers[14] = y; CellOffsets[14] = offsetof(Cell, x); ARRAY_15_MEMBERS
#define ARRAY_15_MEMBERS(x, y) AmountOfArrayMembers[15] = y; CellOffsets[15] = offsetof(Cell, x); ARRAY_16_MEMBERS
#define ARRAY_16_MEMBERS(x, y) AmountOfArrayMembers[16] = y; CellOffsets[16] = offsetof(Cell, x); ARRAY_17_MEMBERS
#define ARRAY_17_MEMBERS(x, y) AmountOfArrayMembers[17] = y; CellOffsets[17] = offsetof(Cell, x); ARRAY_18_MEMBERS
#define ARRAY_18_MEMBERS(x, y) AmountOfArrayMembers[18] = y; CellOffsets[18] = offsetof(Cell, x); ARRAY_19_MEMBERS
#define ARRAY_19_MEMBERS(x, y) AmountOfArrayMembers[19] = y; CellOffsets[19] = offsetof(Cell, x); ARRAY_20_MEMBERS
#define ARRAY_20_MEMBERS(x, y) AmountOfArrayMembers[20] = y; CellOffsets[20] = offsetof(Cell, x); ARRAY_21_MEMBERS
#define ARRAY_21_MEMBERS(x, y) AmountOfArrayMembers[21] = y; CellOffsets[21] = offsetof(Cell, x); ARRAY_22_MEMBERS
#define ARRAY_22_MEMBERS(x, y) AmountOfArrayMembers[22] = y; CellOffsets[22] = offsetof(Cell, x); ARRAY_23_MEMBERS
#define ARRAY_23_MEMBERS(x, y) AmountOfArrayMembers[23] = y; CellOffsets[23] = offsetof(Cell, x); ARRAY_24_MEMBERS
#define ARRAY_24_MEMBERS(x, y) AmountOfArrayMembers[24] = y; CellOffsets[24] = offsetof(Cell, x); ARRAY_25_MEMBERS
#define ARRAY_25_MEMBERS(x, y) AmountOfArrayMembers[25] = y; CellOffsets[25] = offsetof(Cell, x); ARRAY_26_MEMBERS
#define ARRAY_26_MEMBERS(x, y) AmountOfArrayMembers[26] = y; CellOffsets[26] = offsetof(Cell, x); ARRAY_27_MEMBERS
#define ARRAY_27_MEMBERS(x, y) AmountOfArrayMembers[27] = y; CellOffsets[27] = offsetof(Cell, x); ARRAY_28_MEMBERS
#define ARRAY_28_MEMBERS(x, y) AmountOfArrayMembers[28] = y; CellOffsets[28] = offsetof(Cell, x); ARRAY_29_MEMBERS
#define ARRAY_29_MEMBERS(x, y) AmountOfArrayMembers[29] = y; CellOffsets[29] = offsetof(Cell, x); ARRAY_30_MEMBERS
#define ARRAY_30_MEMBERS(x, y) AmountOfArrayMembers[30] = y; CellOffsets[30] = offsetof(Cell, x); ARRAY_31_MEMBERS
#define ARRAY_31_MEMBERS(x, y) AmountOfArrayMembers[31] = y; CellOffsets[31] = offsetof(Cell, x); ARRAY_32_MEMBERS
#define ARRAY_32_MEMBERS(x, y) AmountOfArrayMembers[32] = y; CellOffsets[32] = offsetof(Cell, x); ARRAY_33_MEMBERS
#define ARRAY_33_MEMBERS(x, y) AmountOfArrayMembers[33] = y; CellOffsets[33] = offsetof(Cell, x); ARRAY_34_MEMBERS
#define ARRAY_34_MEMBERS(x, y) AmountOfArrayMembers[34] = y; CellOffsets[34] = offsetof(Cell, x); ARRAY_35_MEMBERS
#define ARRAY_35_MEMBERS(x, y) AmountOfArrayMembers[35] = y; CellOffsets[35] = offsetof(Cell, x); ARRAY_36_MEMBERS
#define ARRAY_36_MEMBERS(x, y) AmountOfArrayMembers[36] = y; CellOffsets[36] = offsetof(Cell, x); ARRAY_37_MEMBERS
#define ARRAY_37_MEMBERS(x, y) AmountOfArrayMembers[37] = y; CellOffsets[37] = offsetof(Cell, x); ARRAY_38_MEMBERS
#define ARRAY_38_MEMBERS(x, y) AmountOfArrayMembers[38] = y; CellOffsets[38] = offsetof(Cell, x); ARRAY_39_MEMBERS
#define ARRAY_39_MEMBERS(x, y) AmountOfArrayMembers[39] = y; CellOffsets[39] = offsetof(Cell, x); ARRAY_40_MEMBERS
#define ARRAY_40_MEMBERS(x, y) AmountOfArrayMembers[40] = y; CellOffsets[40] = offsetof(Cell, x); ARRAY_41_MEMBERS
#define ARRAY_41_MEMBERS(x, y) AmountOfArrayMembers[41] = y; CellOffsets[41] = offsetof(Cell, x); ARRAY_42_MEMBERS
#define ARRAY_42_MEMBERS(x, y) AmountOfArrayMembers[42] = y; CellOffsets[42] = offsetof(Cell, x); ARRAY_43_MEMBERS
#define ARRAY_43_MEMBERS(x, y) AmountOfArrayMembers[43] = y; CellOffsets[43] = offsetof(Cell, x); ARRAY_44_MEMBERS
#define ARRAY_44_MEMBERS(x, y) AmountOfArrayMembers[44] = y; CellOffsets[44] = offsetof(Cell, x); ARRAY_45_MEMBERS
#define ARRAY_45_MEMBERS(x, y) AmountOfArrayMembers[45] = y; CellOffsets[45] = offsetof(Cell, x); ARRAY_46_MEMBERS
#define ARRAY_46_MEMBERS(x, y) AmountOfArrayMembers[46] = y; CellOffsets[46] = offsetof(Cell, x); ARRAY_47_MEMBERS
#define ARRAY_47_MEMBERS(x, y) AmountOfArrayMembers[47] = y; CellOffsets[47] = offsetof(Cell, x); ARRAY_48_MEMBERS
#define ARRAY_48_MEMBERS(x, y) AmountOfArrayMembers[48] = y; CellOffsets[48] = offsetof(Cell, x); ARRAY_49_MEMBERS
#define ARRAY_49_MEMBERS(x, y) AmountOfArrayMembers[49] = y; CellOffsets[49] = offsetof(Cell, x); ARRAY_50_MEMBERS
#define ARRAY_50_MEMBERS(x, y) AmountOfArrayMembers[50] = y; CellOffsets[50] = offsetof(Cell, x); ARRAY_51_MEMBERS
#define ARRAY_51_MEMBERS(x, y) AmountOfArrayMembers[51] = y; CellOffsets[51] = offsetof(Cell, x); ARRAY_52_MEMBERS
#define ARRAY_52_MEMBERS(x, y) AmountOfArrayMembers[52] = y; CellOffsets[52] = offsetof(Cell, x); ARRAY_53_MEMBERS
#define ARRAY_53_MEMBERS(x, y) AmountOfArrayMembers[53] = y; CellOffsets[53] = offsetof(Cell, x); ARRAY_54_MEMBERS
#define ARRAY_54_MEMBERS(x, y) AmountOfArrayMembers[54] = y; CellOffsets[54] = offsetof(Cell, x); ARRAY_55_MEMBERS
#define ARRAY_55_MEMBERS(x, y) AmountOfArrayMembers[55] = y; CellOffsets[55] = offsetof(Cell, x); ARRAY_56_MEMBERS
#define ARRAY_56_MEMBERS(x, y) AmountOfArrayMembers[56] = y; CellOffsets[56] = offsetof(Cell, x); ARRAY_57_MEMBERS
#define ARRAY_57_MEMBERS(x, y) AmountOfArrayMembers[57] = y; CellOffsets[57] = offsetof(Cell, x); ARRAY_58_MEMBERS
#define ARRAY_58_MEMBERS(x, y) AmountOfArrayMembers[58] = y; CellOffsets[58] = offsetof(Cell, x); ARRAY_59_MEMBERS
#define ARRAY_59_MEMBERS(x, y) AmountOfArrayMembers[59] = y; CellOffsets[59] = offsetof(Cell, x); ARRAY_60_MEMBERS
#define ARRAY_60_MEMBERS(x, y) AmountOfArrayMembers[60] = y; CellOffsets[60] = offsetof(Cell, x); ARRAY_61_MEMBERS
#define ARRAY_61_MEMBERS(x, y) AmountOfArrayMembers[61] = y; CellOffsets[61] = offsetof(Cell, x); ARRAY_62_MEMBERS
#define ARRAY_62_MEMBERS(x, y) AmountOfArrayMembers[62] = y; CellOffsets[62] = offsetof(Cell, x); ARRAY_63_MEMBERS
#define ARRAY_63_MEMBERS(x, y) AmountOfArrayMembers[63] = y; CellOffsets[63] = offsetof(Cell, x); ARRAY_64_MEMBERS

#define GENERATE_CELL_STRUCTURE_WITH_SUPPORT_FUNCTIONS(seq) \
    public: \
        struct Cell { CAT(INSERT_MEMBER_RECURSIVE_1 seq, _END) }; \
    protected: \
        static const size_t AmountOfCellMembers = CAT(COUNT_MEMBERS seq, _END); \
        size_t AmountOfArrayMembers[AmountOfCellMembers] = {0}; \
        size_t CellOffsets[AmountOfCellMembers] = {0}; \
        void InitializeArrays() { \
            if(!AmountOfArrayMembers[0]) { \
                CAT(ARRAY_0_MEMBERS seq, _END) \
            } \
        } \
    public: \
        virtual const std::type_info& getDataTypeid() const override {return typeid(STRUCT_DATA_TYPE);} \
        virtual size_t getSizeOfDatatype() const override {return sizeof(STRUCT_DATA_TYPE);} \
        virtual size_t getSizeOfDatastruct() const override {return sizeof(Cell);} \
        virtual size_t getNumberOfElements() const override {return AmountOfCellMembers;} \
        virtual const size_t* getAmountOfArrayMembers() override {InitializeArrays(); return AmountOfArrayMembers;} \
        virtual const size_t* getCellOffsets() override {InitializeArrays(); return CellOffsets;}

#define INSERT_DRAW_PARAM_RECURSIVE_2_END
#define INSERT_DRAW_PARAM_RECURSIVE_1_END
#define INSERT_DRAW_PARAM_RECURSIVE_2(x, y) \
    drawParams.push_back(VisualizationProperty(x, CellOffsets[y], AmountOfArrayMembers[y], \
        data_type_index)); INSERT_DRAW_PARAM_RECURSIVE_1
#define INSERT_DRAW_PARAM_RECURSIVE_1(x, y) \
    drawParams.push_back(VisualizationProperty(x, CellOffsets[y], AmountOfArrayMembers[y], \
        data_type_index)); INSERT_DRAW_PARAM_RECURSIVE_2

#define REGISTER_VISUALIZATION_PARAMETERS(seq) \
    protected: \
        std::vector<VisualizationProperty> drawParams; \
        void InitializeDrawParams() { \
            if(drawParams.empty()) { \
                std::type_index data_type_index(typeid(STRUCT_DATA_TYPE)); \
                CAT(INSERT_DRAW_PARAM_RECURSIVE_1 seq, _END) \
            } \
        } \
    public: \
        virtual const std::vector<VisualizationProperty>* getDrawParams() override { \
            InitializeDrawParams(); \
            return &drawParams; \
        }


#endif /* CELLCONSTRACTION_H */
