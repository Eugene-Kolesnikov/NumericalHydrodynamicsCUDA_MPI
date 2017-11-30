#include "fieldopengl.hpp"

FieldOpenGL::FieldOpenGL(QWidget *parent):
    QOpenGLWidget(parent)
{
}

FieldOpenGL::~FieldOpenGL()
{
    delete format;
    delete m_context;
}

void FieldOpenGL::initialize()
{
    initializeGL();
}

void FieldOpenGL::setFieldPtr(byte* _field)
{
    field = _field;
}

void FieldOpenGL::setSizeOfDatatype(size_t _size_of_datatype)
{
    size_of_datatype = _size_of_datatype;
}

void FieldOpenGL::setScale(size_t _N_X, size_t _N_Y)
{
    N_X = _N_X;
    N_Y = _N_Y;
    _dataSize = N_X * N_Y;
}

void FieldOpenGL::renderFrame()
{
    paintGL();
}

void FieldOpenGL::createMeshPositionVBO()
{
    double xlim[2] = {-0.9f, 0.9f};
    double ylim[2] = {-0.9f, 0.9f};

    double dx = (xlim[1] - xlim[0]) / (N_X-1);
    double dy = (ylim[1] - ylim[0]) / (N_Y-1);

    double* field_xy = new double[2 * _dataSize];

    size_t id = 0;
    /// Generate the mesh
    for(size_t j = 0; j < N_Y; ++j) {
        for(size_t i = 0; i < N_X; ++i) {
            id = 2*(j * N_X + i);
            field_xy[id] = xlim[0] + i*dx;
            field_xy[id + 1] = ylim[0] + j*dy;
        }
    }

    /// Generate the array buffer and upload the mesh to the GPU
    glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, 2 * _dataSize * sizeof(double), field_xy, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    delete[] field_xy;

    glGenBuffers(1, &_fieldValBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void FieldOpenGL::createMeshIndexBuffer()
{
    int size = (N_X - 1) * (N_Y - 1) * 6 * sizeof(GLuint);
    _index_buf_count = (N_X - 1) * (N_Y - 1) * 6;

    GLuint* ids = new GLuint[_index_buf_count];
    size_t global;

    for(size_t j = 0; j < N_Y-1; ++j) {
        for(size_t i = 0; i < N_X-1; ++i) {
            global = 6*(j*(N_X-1)+i);
            ids[global + 0] = j*N_X+i;
            ids[global + 1] = j*N_X+i+1;
            ids[global + 2] = (j+1)*N_X+i;
            ids[global + 3] = (j+1)*N_X+i;
            ids[global + 4] = (j+1)*N_X+i+1;
            ids[global + 5] = j*N_X+i+1;
        }
    }

    /// Generate the index buffer
    glGenBuffers(1, &_indexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, ids, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    delete[] ids;
}

void FieldOpenGL::initializeGL()
{
    glClearColor(0.0f, 0.0f, 0.1f, 1.0f);
    /// Generate the Vertex Array Buffer
    glGenVertexArrays(1, &_vao);
    glBindVertexArray(_vao);
    /// Generate the field mesh and the index mesh
    createMeshPositionVBO();
    createMeshIndexBuffer();
    /// Load and compile shaders
    _programID = compileShaders();
}

void FieldOpenGL::paintGL()
{
    updateField();
    /// Bind buffers, shader program, vertex attribute pointers
    glBindVertexArray(_vao);
    glUseProgram(_programID);
    /// Bind the buffer of XY-field
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glEnableVertexAttribArray(0);
    /// Attrib pointer for (x,y) coordinates
    glVertexAttribPointer(
        0, // attribute 0 -- the layout in the shader.
        2, // size
        GL_FLOAT, // type
        GL_FALSE, // normalized?
        0, // stride
        (void*)0 // array buffer offset
    );
    /// Bind the buffer of the field's property (ex: density)
    glBindBuffer(GL_ARRAY_BUFFER, _fieldValBuffer);
    glEnableVertexAttribArray(1);
    /// Attrib pointer for the field's property
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);
    /// Bind the element array buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexBuffer);
    /// Render the field using the element array
    glDrawElements(GL_TRIANGLES, _index_buf_count, GL_UNSIGNED_INT, 0);
    /// Unbind buffers, pointers, shaders
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glUseProgram(0);
    glBindVertexArray(0);
    /// Swap buffers
    //m_context->swapBuffers(this);
    //m_context->doneCurrent();
}

void FieldOpenGL::updateField()
{
    glBindBuffer(GL_ARRAY_BUFFER, _fieldValBuffer);
    glBufferData(GL_ARRAY_BUFFER, _dataSize * size_of_datatype, field, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

GLuint FieldOpenGL::compileShaders()
{
    /// Generate new shader indeces
    GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

    GLint Result = GL_FALSE;
    int InfoLogLength;

    /// Read and compile the Vertex Shader
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

    /// Read and compile the Fragment Shader
    std::string FragmentShaderCode = getFragmentShader();
    char const * FragmentSourcePointer = FragmentShaderCode.c_str();
    glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL); // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    glCompileShader(FragmentShaderID);

    /// Check if the compilation of the Fragment Shader was successful
    glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength > 0) {
            std::vector<char> FragmentShaderErrorMessage(InfoLogLength+1);
            glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
            throw std::runtime_error(&FragmentShaderErrorMessage[0]);
    }

    /// Link the compiled shaders to the program
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

/**
 * @brief Get the vertex shader in the form of an std::sting.
 * @return The string which contains the content of a vertex shader.
*/
std::string FieldOpenGL::getVertexShader()
{
    return
    "#version 410 core \n"
    "layout(location = 0) in vec2 pos; \n"
    "layout(location = 1) in float field_val; \n"
    "out float field; \n"
    "void main() { \n"
    "   gl_Position = vec4(pos.x, pos.y, 0.0f, 1.0); \n"
    "   field = field_val; \n"
    "} \n";
}

/**
 * @brief Get the fragment shader in the form of an std::sting.
 * @return The string which contains the content of a fragment shader.
*/
std::string FieldOpenGL::getFragmentShader()
{
    return
    "#version 410 core \n"
    "in float field; \n"
    "vec3 colorbar(float field) \n"
    "{ \n"
    "   vec3 color = vec3(0,0,0); \n"
    "   if(field >= 1.0/2.0) { \n"
    "       float d = field - 1.0/2.0; \n"
    "       color = vec3(0,(1.0/2.0-d)*2.0,d*2.0); \n"
    "   } else { \n"
    "       float d = 1.0/2.0 - field; \n"
    "       color = vec3(d*2.0,(1.0/2.0-d)*2.0,0); \n"
    "   } \n"
    "   return color; \n"
    "} \n"
    "void main (void) { \n"
    "   gl_FragColor = vec4(colorbar(field),1); \n"
    "} \n";
}
