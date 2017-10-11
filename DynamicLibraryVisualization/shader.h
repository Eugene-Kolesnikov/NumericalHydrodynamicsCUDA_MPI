/* 
 * File:   shader.h
 * Author: eugene
 *
 * Created on October 8, 2017, 11:55 PM
 */

#ifndef SHADERS_H
#define SHADERS_H

#include <OpenGL/gl3.h>
#include <string>

std::string loadShaderFile(const std::string& filename);
GLuint compileShaders(std::string vertexPath, std::string fragmentPath);

#endif /* SHADERS_H */

