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
 * @brief Reads the shader file.
 * @param filename -- The path to the shader file in the system.
 * @return The string which contains the content of a shader file.
*/
std::string loadShaderFile(const std::string& filename);

/**
 * @brief Reads the shader file.
 * @param vertexPath -- The path to the vertex shader file in the system.
 * @param fragmentPath -- The path to the fragment shader file in the system.
 * @return The ID of a compiled and linked shader program.
*/
GLuint compileShaders(std::string vertexPath, std::string fragmentPath);

#endif /* SHADERS_H */

