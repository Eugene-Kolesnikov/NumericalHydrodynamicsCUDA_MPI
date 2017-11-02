/**
* @file shader.cpp
* @brief This file contains description of functions which load, compile,
 * and link shader files to the program.
* @author Eugene Kolesnikov
* @date 8/10/2017
*/

#include <exception>
#include <vector>
#include <fstream>
#include "shader.h"

/**
 * @brief Get the vertex shader in the form of an std::sting.
 * @return The string which contains the content of a vertex shader.
*/
std::string getVertexShader()
{
    return
    "#version 410 core"
    "layout(location = 0) in vec2 pos;"
    "layout(location = 1) in float field_val;"
    "out float field;"
    "void main() {"
    "   gl_Position = vec4(pos.x, pos.y, 0.0f, 1.0);"
    "   field = field_val;"
    "}";
}

/**
 * @brief Get the fragment shader in the form of an std::sting.
 * @return The string which contains the content of a fragment shader.
*/
std::string getFragmentShader()
{
    return
    "#version 410 core"
    "in float field;"
    "out vec4 fColor;"
    "vec3 colorbar(float field)"
    "{"
    "   vec3 color = vec3(0,0,0);"
    "   if(field >= 1.0/2.0) {"
    "       float d = field - 1.0/2.0;"
    "       color = vec3(0,(1.0/2.0-d)*2.0,d*2.0);"
    "   } else {"
    "       float d = 1.0/2.0 - field;"
    "       color = vec3(d*2.0,(1.0/2.0-d)*2.0,0);"
    "   }"
    "   return color;"
    "}"
    "void main (void) {"
    "   fColor = vec4(colorbar(field),1);"
    "}";
}

/**
 * @brief Reads the shader file, which in this version of the program
  * is included in the program in the form of a char* string.
 * @return The ID of a compiled and linked shader program.
*/
GLuint compileShaders()
{
    /// Generate new shader indeces
    GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

    GLint Result = GL_FALSE;
	int InfoLogLength;

    /// Read and compile the Vertex Shader
    #ifdef __DEBUG__
	   printf("Compiling shader : %s\n", vertexPath.c_str());
    #endif
    std::string VertexShaderCode = getVertexShader();
	char const* VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
	glCompileShader(VertexShaderID);

	/// Check if the compilation of the Vertex Shader was successful
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0){
		std::vector<char> VertexShaderErrorMessage(InfoLogLength+1);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		throw std::runtime_error(&VertexShaderErrorMessage[0]);
	}

    // Read and compile the Fragment Shader
    #ifdef __DEBUG__
	   printf("Compiling shader : %s\n", fragmentPath.c_str());
    #endif
    std::string FragmentShaderCode = getFragmentShader();
	char const * FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
	glCompileShader(FragmentShaderID);

	/// Check if the compilation of the Fragment Shader was successful
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0){
            std::vector<char> FragmentShaderErrorMessage(InfoLogLength+1);
            glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
            throw std::runtime_error(&FragmentShaderErrorMessage[0]);
	}

    /// Link the compiled shaders to the program
    #ifdef __DEBUG__
	   printf("Linking program\n");
    #endif
	GLuint _shaderProgram = glCreateProgram();
	glAttachShader(_shaderProgram, VertexShaderID);
	glAttachShader(_shaderProgram, FragmentShaderID);
	glLinkProgram(_shaderProgram);

	// Check the program
	glGetProgramiv(_shaderProgram, GL_LINK_STATUS, &Result);
	glGetProgramiv(_shaderProgram, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0){
		std::vector<char> ProgramErrorMessage(InfoLogLength+1);
		glGetProgramInfoLog(_shaderProgram, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		throw std::runtime_error(&ProgramErrorMessage[0]);
	}

	return _shaderProgram;
}
