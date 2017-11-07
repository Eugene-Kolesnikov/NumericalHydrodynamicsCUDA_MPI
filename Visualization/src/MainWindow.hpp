/**
* @file MainWindow.hpp
* @brief This header file contains the MainWindow class with an OpenGL context. 
* @author Eugene Kolesnikov 
* @date 8/10/2017 
*/

#ifndef MAINWINDOW_HPP
#define MAINWINDOW_HPP

#include <string>
#include <GLFW/glfw3.h>

/**
* @class MainWindow
* @brief The class which is responsible for all necessary functions
 * related to the window itself: creation, termination, events handling,
 * cleaning.
* @author Eugene Kolesnikov 
* @date 8/10/2017 
*/
class MainWindow
{
public:
    MainWindow();
    ~MainWindow();

    /**
     * @brief Creates the window with a specified title.
     * @param title -- the title of the window.
    */
    void create(std::string title);
    
    /**
     * @brief Returns the pointer to the GLFW3 class which stores the window.
     * @return pointer to the GLFW3 class.
    */
    GLFWwindow* window() { 
        return _window; 
    }
    
    /**
     * @brief Terminates the window and clears the memory.
    */
    void terminate() { 
        glfwTerminate(); 
    }
    
    /**
     * @brief Sets up the background color of the window.
     * @param r -- red component of the RGBA color.
     * @param g -- blue component of the RGBA color.
     * @param b -- green component of the RGBA color.
     * @param a -- alpha component of the RGBA color.
    */
    void setBGColor(float r, float g, float b, float a) { 
        glClearColor(r, g, b, a); 
    }
    
    /**
     * @brief Clears the window from previously rendered image.
    */
    void clearScreen() { 
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
    }

    /** 
     * @brief Function which is called each time a GLFW error occurs.
     * @param error -- error code.
     * @param description -- human-readable description.
    */
    static void error_callback(int error, const char* description);
    
    /** 
     * @brief Key event function which notifies when a physical key is pressed or released or when it repeats
     * @param window -- The window that received the event.
     * @param key -- The keyboard key that was pressed or released.
     * @param scancode -- The system-specific scancode of the key.
     * @param action -- GLFW_PRESS, GLFW_RELEASE or GLFW_REPEAT.
     * @param mods -- Bit field describing which modifier keys were held down: 
      * GLFW_MOD_ALT, GLFW_MOD_CONTROL, GLFW_MOD_SHIFT, or GLFW_MOD_SUPER.
    */
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    
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
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    
    /** 
     * @brief mouse button event function which notifies when the cursor moves over the window.
      * The callback functions receives the cursor position, measured in screen 
      * coordinates but relative to the top-left corner of the window client area.
     * @param x -- x-coordinate.
     * @param y -- y-coordinate.
    */
    static void cursor_position_callback(GLFWwindow* window, double x, double y);
    
protected:  
    /**
     * @brief This method returns the resolution of the 
      * screen to open the window of an appropriate size.
     * @return minimum of the height-width resolution of the screen in pixels.
    */
    unsigned short get_resolution();

protected:
    GLFWwindow* _window; /**< GLFW3 class which stores the window on which the
                           * OpenGL renders the field. */
};

#endif /* MAINWINDOW_HPP */

