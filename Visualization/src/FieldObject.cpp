/**
* @file FieldObject.cpp
* @brief This file contains descriptions of functions (FieldObject class methods) which are responsible
 * for rendering the field and storing all necessary environmental variables.
* @author Eugene Kolesnikov
* @date 8/10/2017
*/

#include "FieldObject.hpp"
#include "shader.h"
#include <exception>
#include "../../ComputationalScheme/src/cell.h"

extern GLuint programID;

FieldObject::FieldObject(std::string path)
{
    _data = nullptr;
    _output_option = PNG;
    outputPath = path;
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

/**
 * @brief Initializes buffer objects, generates the field mesh, index mesh and
  * uploads it to the GPU, compiles shaders.
 * @param N_X -- amount of grid cells along the X direction.
 * @param N_Y -- amount of grid cells along the Y direction.
*/
void FieldObject::init(size_t N_X, size_t N_Y)
{
    _dataSize = N_X * N_Y;
    _data = new GLfloat[_dataSize];

    /// Generate the Vertex Array Buffer
    glGenVertexArrays(1, &_vao);
    glBindVertexArray(_vao);

    // Generate the field mesh and the index mesh
    createMeshPositionVBO(N_X, N_Y);
    createMeshIndexBuffer(N_X, N_Y);

    /// Load and compile shaders
    _programID = compileShaders();

    init_output(_output_option, outputPath.c_str());
}

/**
 * @brief Renders the field.
 * @param field -- pointer to the field.
 * @param N_X -- amount of grid cells along the X direction.
 * @param N_Y -- amount of grid cells along the Y direction.
*/
void FieldObject::render(void* field, size_t N_X, size_t N_Y)
{
    /// Update the field information
    updateField(field, N_X, N_Y);

    /// Bind buffers, shader program, vertex attribute pointers
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

    /// Render the field
    glDrawElements(GL_TRIANGLES, _index_buf_count, GL_UNSIGNED_INT, 0);

    /// Save the rendered image to the file of the output style
    writeframe_output(_output_option, outputPath.c_str());

    /// Unbind buffers, pointers, shaders
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glUseProgram(0);
    glBindVertexArray(0);
}

/**
 * @brief Generates the field mesh [-0.9, 0.9]^2.
 * @param N_X -- amount of grid cells along the X direction.
 * @param N_Y -- amount of grid cells along the Y direction.
*/
void FieldObject::createMeshPositionVBO(size_t N_X, size_t N_Y)
{
    float xlim[2] = {-0.9f, 0.9f};
    float ylim[2] = {-0.9f, 0.9f};

    float dx = (xlim[1] - xlim[0]) / (N_X-1);
    float dy = (ylim[1] - ylim[0]) / (N_Y-1);

    glm::vec2* field_xy = new glm::vec2[_dataSize];

    /// Generate the mesh
    for(int j = 0; j < N_Y; ++j) {
        for(int i = 0; i < N_X; ++i) {
            field_xy[j*N_X+i] = glm::vec2(xlim[0] + i*dx, ylim[0] + j*dy);
        }
    }

    /// Generate the array buffer and upload the mesh to the GPU
    glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, _dataSize * sizeof(glm::vec2), field_xy, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    delete[] field_xy;


    /// Generate the empty array buffer of the field information and upload to the GPU
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

/**
 * @brief Generates the index mesh for rendering.
 * @param N_X -- amount of grid cells along the X direction.
 * @param N_Y -- amount of grid cells along the Y direction.
*/
void FieldObject::createMeshIndexBuffer(size_t N_X, size_t N_Y)
{
    int size = (N_X - 1) * (N_Y - 1) * 6 * sizeof(GLuint);
    _index_buf_count = (N_X - 1) * (N_Y - 1) * 6;

    /// Generate the index buffer
    glGenBuffers(1, &_indexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

    GLuint* indices = (GLuint*)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);
    if (!indices) {
        throw std::runtime_error("Failed to create an index buffer.");
    }

    /// Generate the index mesh and automatically upload to the GPU
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

/**
 * @brief Re-uploads updated field characteristics (ex. density, velocity, energy).
 * @param field -- pointer to the field.
 * @param N_X -- amount of grid cells along the X direction.
 * @param N_Y -- amount of grid cells along the Y direction.
*/
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

/**
 * @brief Set up the a type of the output: PPM, PNG, MPEG
 * @param outOption -- the type of the output.
*/
void FieldObject::setOutputOption(enum OUTPUT_OPTION outOption)
{
    _output_option = outOption;
}
