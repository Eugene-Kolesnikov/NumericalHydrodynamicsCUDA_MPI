/**
* @file MainWindow.cpp
* @brief This file contains descriptions of functions (MainWindow class methods) which are responsible for all necessary functions
 * related to the window itself: creation, termination, events handling, cleaning.
* @author Eugene Kolesnikov 
* @date 8/10/2017 
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

/**
 * @brief This method returns the resolution of the 
  * screen to open the window of an appropriate size.
 * @return minimum of the height-width resolution of the screen in pixels.
*/
unsigned short MainWindow::get_resolution() 
{
    const GLFWvidmode * mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    return (mode->width > mode->height ? mode->height : mode->width);
}

/**
 * @brief Creates the window with a specified title.
 * @param title -- the title of the window.
*/
void MainWindow::create(std::string title)
{
    /// Initialize GLFW
    if( !glfwInit() ) {
        throw std::runtime_error("Failed to initialize GLFW.");
    }

    /// set up the OpenGL 4.1 context
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    /// set up the screen resolution
    unsigned short size_wh = 0.49 * get_resolution();
    set_output_windowsize(size_wh);
    
    /// Open a window and create its OpenGL context
    _window = glfwCreateWindow(size_wh, size_wh, title.c_str(), NULL, NULL);
    if( _window == NULL ){
        glfwTerminate();
        throw std::runtime_error("Failed to open GLFW window.");
    }
    glfwMakeContextCurrent(_window);

    /// Ensure we can capture the escape key being pressed below
    glfwSetInputMode(_window, GLFW_STICKY_KEYS, GL_TRUE);

    glEnable(GL_DEPTH_TEST); /// Enable the depth-test
    glDepthFunc(GL_LEQUAL); /// Use LEQUAL function for the depth-test
    glEnable(GL_BLEND); /** allow blending (Blending is the stage of OpenGL 
     * rendering pipeline that takes the fragment color outputs from the 
     * Fragment Shader and combines them with the colors in the color buffers 
     * that these outputs map to)
    */
}

/** 
 * @brief Function which is called each time a GLFW error occurs.
 * @param error -- error code.
 * @param description -- human-readable description.
*/
void MainWindow::error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

/** 
 * @brief Key event function which notifies when a physical key is pressed or released or when it repeats
 * @param window -- The window that received the event.
 * @param key -- The keyboard key that was pressed or released.
 * @param scancode -- The system-specific scancode of the key.
 * @param action -- GLFW_PRESS, GLFW_RELEASE or GLFW_REPEAT.
 * @param mods -- Bit field describing which modifier keys were held down: 
  * GLFW_MOD_ALT, GLFW_MOD_CONTROL, GLFW_MOD_SHIFT, or GLFW_MOD_SUPER.
*/
void MainWindow::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    /// Not Implemented
}

/** 
 * @brief mouse button event function which notifies when a mouse button is pressed or released
 * @param window -- The window that received the event..
 * @param button -- The mouse button that was pressed or released: 
  * GLFW_MOUSE_BUTTON_1, ..., GLFW_MOUSE_BUTTON_8, GLFW_MOUSE_BUTTON_LAST, GLFW_MOUSE_BUTTON_LEFT,
  * GLFW_MOUSE_BUTTON_MIDDLE, GLFW_MOUSE_BUTTON_RIGHT.
 * @param action -- One of GLFW_PRESS or GLFW_RELEASE.
 * @param mods -- Bit field describing which modifier keys were held down: 
  * GLFW_MOD_ALT, GLFW_MOD_CONTROL, GLFW_MOD_SHIFT, or GLFW_MOD_SUPER.
*/
void MainWindow::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    /// Not Implemented
}

/** 
 * @brief mouse button event function which notifies when the cursor moves over the window.
  * The callback functions receives the cursor position, measured in screen 
  * coordinates but relative to the top-left corner of the window client area.
 * @param x -- x-coordinate.
 * @param y -- y-coordinate.
*/
void MainWindow::cursor_position_callback(GLFWwindow* window, double x, double y)
{
    /// Not Implemented
}


