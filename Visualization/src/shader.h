/**
* @file shader.h
* @brief This header file contains the interface to functions which load, compile,
 * and link shader files to the program.
* @author Eugene Kolesnikov
* @date 8/10/2017
*/

#ifndef SHADERS_H
#define SHADERS_H

#include <OpenGL/gl3.h>
#include <string>

/**
 * @brief Get the vertex shader in the form of an std::sting.
 * @return The string which contains the content of a vertex shader.
*/
std::string getVertexShader();

/**
 * @brief Get the fragment shader in the form of an std::sting.
 * @return The string which contains the content of a fragment shader.
*/
std::string getFragmentShader();

/**
 * @brief Reads the shader file, which in this version of the program
  * is stored in the program in the form of a char* string.
 * @return The ID of a compiled and linked shader program.
*/
GLuint compileShaders();

#endif /* SHADERS_H */
