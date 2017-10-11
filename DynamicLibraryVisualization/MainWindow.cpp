/* 
 * File:   MainWindow.cpp
 * Author: eugene
 * 
 * Created on October 8, 2017, 11:50 PM
 */

#include "MainWindow.hpp"
#include <exception>
#include <OpenGL/gl3.h>
#include "offscreen.h"


MainWindow::MainWindow()
{
    _window = 0;
}

MainWindow::~MainWindow()
{
    glfwTerminate();
}

unsigned short MainWindow::get_resolution() 
{
    const GLFWvidmode * mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    return (mode->width > mode->height ? mode->height : mode->width);
}

void MainWindow::create(std::string title)
{
    // Initialize GLFW
    if( !glfwInit() ) {
        throw std::runtime_error("Failed to initialize GLFW.");
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // set up the screen resolution
    unsigned short size_wh = 0.49 * get_resolution();
    set_output_windowsize(2 * size_wh);
    
    // Open a window and create its OpenGL context
    _window = glfwCreateWindow(size_wh, size_wh, title.c_str(), NULL, NULL);
    if( _window == NULL ){
        glfwTerminate();
        throw std::runtime_error("Failed to open GLFW window.");
    }
    glfwMakeContextCurrent(_window);

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(_window, GLFW_STICKY_KEYS, GL_TRUE);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_BLEND);
}

void MainWindow::terminate()
{
    glfwTerminate();
}

void MainWindow::setBGColor(float r, float g, float b, float a)
{
    glClearColor(r, g, b, a);
}

void MainWindow::clearScreen()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

// Function is called with an error code and a human-readable description each time a GLFW error occurs
void MainWindow::error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

// "key callback" notifies when a physical key is pressed or released or when it repeats
void MainWindow::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{

}

// "mouse button callback" notifies when a mouse button is pressed or released
void MainWindow::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{

}

// "cursor position callback" notifies when the cursor moves over the window
// The callback functions receives the cursor position, measured in screen
// coordinates but relative to the top-left corner of the window client area.
void MainWindow::cursor_position_callback(GLFWwindow* window, double x, double y)
{

}


