/* 
 * File:   FieldObject.cpp
 * Author: eugene
 * 
 * Created on October 8, 2017, 11:50 PM
 */

#include "FieldObject.hpp"
#include "shader.h"
#include <exception>
#include "cell.h"

extern GLuint programID;

FieldObject::FieldObject()
{
    _data = nullptr;
    _output_option = PNG;
}

FieldObject::~FieldObject()
{
    if(_data != 0)
        delete[] _data;

    glDeleteProgram(_programID);
    glDeleteBuffers(1, &_vbo);
    glDeleteVertexArrays(1, &_vao);
    deinit_output(_output_option);
}

void FieldObject::init(size_t N_X, size_t N_Y)
{
    _dataSize = N_X * N_Y;
    _data = new GLfloat[_dataSize];

    glGenVertexArrays(1, &_vao);
    glBindVertexArray(_vao);

    createMeshPositionVBO(N_X, N_Y);
    createMeshIndexBuffer(N_X, N_Y);

    _programID = compileShaders("/Users/eugene/NetBeansProjects/DynamicLibraryVisualization/shaders/draw_field.vert.glsl",
                                "/Users/eugene/NetBeansProjects/DynamicLibraryVisualization/shaders/draw_field.frag.glsl");
    
    init_output(_output_option);
}

void FieldObject::render(void* field, size_t N_X, size_t N_Y)
{
    updateField(field, N_X, N_Y);
    
    glBindVertexArray(_vao);
    glUseProgram(_programID);
    
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0, // attribute 0 -- the layout in the shader.
	2, // size
	GL_FLOAT, // type
	GL_FALSE, // normalized?
	0, // stride
	(void*)0 // array buffer offset
    );
    
    glBindBuffer(GL_ARRAY_BUFFER, _fieldValBuffer);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexBuffer);

    glDrawElements(GL_TRIANGLES, _index_buf_count, GL_UNSIGNED_INT, 0);
    writeframe_output(_output_option);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glUseProgram(0);
    glBindVertexArray(0);
}

void FieldObject::createMeshPositionVBO(size_t N_X, size_t N_Y)
{
    float xlim[2] = {-0.9f, 0.9f};
    float ylim[2] = {-0.9f, 0.9f};

    float dx = (xlim[1] - xlim[0]) / (N_X-1);
    float dy = (ylim[1] - ylim[0]) / (N_Y-1);

    glm::vec2* field_xy = new glm::vec2[_dataSize];
    
    for(int j = 0; j < N_Y; ++j) {
        for(int i = 0; i < N_X; ++i) {
            field_xy[j*N_X+i] = glm::vec2(xlim[0] + i*dx, ylim[0] + j*dy);
        }
    }

    glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, _dataSize * sizeof(glm::vec2), field_xy, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    delete[] field_xy;
    
    
    // %% CREATION AND INITIALIZATION OF A '_fieldValBuffer'.
    
    for(int j = 0; j < N_Y; ++j) {
        for(int i = 0; i < N_X; ++i) {
            _data[j*N_X+i] = 0.0;
        }
    }
    
    glGenBuffers(1, &_fieldValBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, _fieldValBuffer);
    glBufferData(GL_ARRAY_BUFFER, _dataSize * sizeof(float), _data, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void FieldObject::createMeshIndexBuffer(size_t N_X, size_t N_Y)
{
    int size = (N_X - 1) * (N_Y - 1) * 6 * sizeof(GLuint);
    _index_buf_count = (N_X - 1) * (N_Y - 1) * 6;

    // create index buffer
    glGenBuffers(1, &_indexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

    GLuint* indices = (GLuint*)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);
    if (!indices) {
        throw std::runtime_error("Failed to create an index buffer.");
    }

    for(int j = 0; j < N_Y-1; ++j) {
        for(int i = 0; i < N_X-1; ++i) {
            *indices++ = j*N_X+i;
            *indices++ = j*N_X+i+1;
            *indices++ = (j+1)*N_X+i;

            *indices++ = (j+1)*N_X+i;
            *indices++ = (j+1)*N_X+i+1;
            *indices++ = j*N_X+i+1;
        }
    }

    glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void FieldObject::updateField(void* field, size_t N_X, size_t N_Y)
{
    Cell* FieldCell = (Cell*)field;
    
    for(int j = 0; j < N_Y; ++j) {
        for(int i = 0; i < N_X; ++i) {
            _data[j*N_X+i] = FieldCell[j*N_X+i].r;
        }
    }
    
    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _fieldValBuffer);
    glBufferData(GL_ARRAY_BUFFER, _dataSize * sizeof(GLfloat), _data, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void FieldObject::setOutputOption(enum OUTPUT_OPTION outOption)
{
    _output_option = outOption;
}

