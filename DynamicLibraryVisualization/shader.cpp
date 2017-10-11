#include <exception>
#include <vector>
#include <fstream>
#include "shader.h"

std::string loadShaderFile(const std::string& filename)
{
    std::string shaderCode;
	std::ifstream shaderStream(filename.c_str(), std::ios::in);
	if(shaderStream.is_open()) {
		std::string line = "";
		while(getline(shaderStream, line))
			shaderCode += ("\n" + line);
		shaderStream.close();
	} else {
        throw std::runtime_error("Shader file can't be opened.");
	}
    return shaderCode;
}

GLuint compileShaders(std::string vertexPath, std::string fragmentPath)
{
    GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

    GLint Result = GL_FALSE;
	int InfoLogLength;

    // Compile Vertex Shader
    #ifdef __DEBUG__
	   printf("Compiling shader : %s\n", vertexPath.c_str());
    #endif
    std::string VertexShaderCode = loadShaderFile(vertexPath);
	char const* VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
	glCompileShader(VertexShaderID);

	// Check Vertex Shader
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0){
		std::vector<char> VertexShaderErrorMessage(InfoLogLength+1);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		throw std::runtime_error(&VertexShaderErrorMessage[0]);
	}

    // Compile Fragment Shader
    #ifdef __DEBUG__
	   printf("Compiling shader : %s\n", fragmentPath.c_str());
    #endif
    std::string FragmentShaderCode = loadShaderFile(fragmentPath);
	char const * FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
	glCompileShader(FragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0){
		std::vector<char> FragmentShaderErrorMessage(InfoLogLength+1);
		glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
		throw std::runtime_error(&FragmentShaderErrorMessage[0]);
	}

    // Link the program
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
