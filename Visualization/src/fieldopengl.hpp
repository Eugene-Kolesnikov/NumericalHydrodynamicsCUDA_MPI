#ifndef FIELDOPENGL_H
#define FIELDOPENGL_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions_4_1_Core>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QSurface>
#include <string>

class FieldOpenGL : public QOpenGLWidget
{
    typedef char byte;
public:
    FieldOpenGL(QWidget *parent = 0);
    ~FieldOpenGL();

public:
    void initialize();
    void setFieldPtr(byte* _field);
    void setSizeOfDatatype(size_t _size_of_datatype);
    void setScale(size_t _N_X, size_t _N_Y);
    void renderFrame();

protected:
    void createMeshPositionVBO();
    void createMeshIndexBuffer();
    void updateField();
    GLuint compileShaders();
    std::string getFragmentShader();
    std::string getVertexShader();

protected:
    void initializeGL() override;
    void paintGL() override;

protected:
    QSurfaceFormat* format;
    QOpenGLFunctions_4_1_Core* OpenGL;
    QOpenGLContext *m_context;
    byte* field;
    size_t size_of_datatype;
    size_t N_X;
    size_t N_Y;

protected:
    GLuint _vao; /**< Vertex Array Object -- OpenGL Object that stores all of the state needed to supply vertex data */
    GLuint _vbo; /**< Vertex Buffer Object which stores the 2D field without the field information (ex. density, velocity, energy) */
    GLuint _fieldValBuffer; /**< Vertex Buffer Object which stores the field information
                             * (ex. density, velocity, energy) in points which are stored in _vbo.
                             * _vbo and _fieldValBuffer are split because the first buffer is static and the other one is dynamic. */
    GLuint _indexBuffer; /**< Vertex Buffer Object which stores indeces which are used to render the field */
    size_t _index_buf_count; /**< Number of indeces. */
    size_t _dataSize; /**< Number of points in the generated field mesh. */
    GLuint _programID; /**< Environment with compiled shaders. */
};

#endif // FIELDOPENGL_H
