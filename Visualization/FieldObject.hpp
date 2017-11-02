/**
* @file FieldObject.hpp
* @brief This header file contains the FieldObject class which is responsible 
 * for rendering the field and storing all necessary environmental variables.
* @author Eugene Kolesnikov 
* @date 8/10/2017 
*/

#ifndef FIELDOBJECT_HPP
#define FIELDOBJECT_HPP

#include <OpenGL/gl3.h>
#include <glm/glm.hpp>
#include "offscreen.h"

/**
* @class FieldObject
* @brief The class which is responsible for rendering the field and storing all
 * necessary environmental variables.
* @author Eugene Kolesnikov 
* @date 8/10/2017 
*/
class FieldObject
{
public:
    FieldObject();
    ~FieldObject();

    /**
     * @brief Initializes buffer objects, generates the field mesh, index mesh and
      * uploads it to the GPU, compiles shaders.
     * @param N_X -- amount of grid cells along the X direction.
     * @param N_Y -- amount of grid cells along the Y direction.
    */
    void init(size_t N_X, size_t N_Y);
    
    /**
     * @brief Renders the field.
     * @param field -- pointer to the field.
     * @param N_X -- amount of grid cells along the X direction.
     * @param N_Y -- amount of grid cells along the Y direction.
    */
    void render(void* field, size_t N_X, size_t N_Y);
    
    /**
     * @brief Set up the a type of the output: PPM, PNG, MPEG
     * @param outOption -- the type of the output.
    */
    void setOutputOption(enum OUTPUT_OPTION outOption);

protected:
    /**
     * @brief Generates the field mesh [-0.9, 0.9]^2.
     * @param N_X -- amount of grid cells along the X direction.
     * @param N_Y -- amount of grid cells along the Y direction.
    */
    void createMeshPositionVBO(size_t N_X, size_t N_Y);
    
    /**
     * @brief Generates the index mesh for rendering.
     * @param N_X -- amount of grid cells along the X direction.
     * @param N_Y -- amount of grid cells along the Y direction.
    */
    void createMeshIndexBuffer(size_t N_X, size_t N_Y);
    
    /**
     * @brief Re-uploads updated field characteristics (ex. density, velocity, energy).
     * @param field -- pointer to the field.
     * @param N_X -- amount of grid cells along the X direction.
     * @param N_Y -- amount of grid cells along the Y direction.
    */
    void updateField(void* field, size_t N_X, size_t N_Y);

protected:
    GLuint _vao; /**< Vertex Array Object -- OpenGL Object that stores all of the state needed to supply vertex data */
    GLuint _vbo; /**< Vertex Buffer Object which stores the 2D field without the field information (ex. density, velocity, energy) */
    GLuint _fieldValBuffer; /**< Vertex Buffer Object which stores the field information 
                             * (ex. density, velocity, energy) in points which are stored in _vbo.
                             * _vbo and _fieldValBuffer are split because the first buffer is static and the other one is dynamic. */
    GLuint _indexBuffer; /**< Vertex Buffer Object which stores indeces which are used to render the field */
    size_t _index_buf_count; /**< Number of indeces. */
    GLfloat* _data; /**< Pointer to an array which stores the information (ex. density, velocity, energy) of the generated field mesh. */
    size_t _dataSize; /**< Number of points in the generated field mesh. */
    GLuint _programID; /**< Environment with compiled shaders. */
    enum OUTPUT_OPTION _output_option; /**< Type of the output: save in an image (PPM, PNG) or generate a video file (MPEG). */
};

#endif /* FIELDOBJECT_HPP */

