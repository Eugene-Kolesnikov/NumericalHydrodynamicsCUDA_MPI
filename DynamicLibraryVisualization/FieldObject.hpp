/* 
 * File:   FieldObject.hpp
 * Author: eugene
 *
 * Created on October 8, 2017, 11:50 PM
 */

#ifndef FIELDOBJECT_HPP
#define FIELDOBJECT_HPP

#include <OpenGL/gl3.h>
#include <glm/glm.hpp>
#include "offscreen.h"

class FieldObject
{
public:
    FieldObject();
    ~FieldObject();

    void init(size_t N_X, size_t N_Y);
    void render(void* field, size_t N_X, size_t N_Y);
    void setOutputOption(enum OUTPUT_OPTION outOption);

protected:
    void createMeshPositionVBO(size_t N_X, size_t N_Y);
    void createMeshIndexBuffer(size_t N_X, size_t N_Y);
    void updateField(void* field, size_t N_X, size_t N_Y);

protected:
    GLuint _vao;
    GLuint _vbo; // storage of 2D field without the field information
    GLuint _fieldValBuffer; // buffer of value of the field in points which are stored in _vbo
    GLuint _indexBuffer;
    size_t _index_buf_count;
    GLfloat* _data;
    size_t _dataSize;
    GLuint _programID;
    enum OUTPUT_OPTION _output_option;
};

#endif /* FIELDOBJECT_HPP */

