/* 
 * File:   MainWindow.hpp
 * Author: eugene
 *
 * Created on October 8, 2017, 11:50 PM
 */

#ifndef MAINWINDOW_HPP
#define MAINWINDOW_HPP

#include <string>
#include <GLFW/glfw3.h>

class MainWindow
{
public:
    MainWindow();
    ~MainWindow();

    void create(std::string title);
    GLFWwindow* window() { return _window; }
    void terminate();
    void setBGColor(float r, float g, float b, float a);
    void clearScreen();

    static void error_callback(int error, const char* description);
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void cursor_position_callback(GLFWwindow* window, double x, double y);
    
protected:
    unsigned short get_resolution();

protected:
    GLFWwindow* _window;
};

#endif /* MAINWINDOW_HPP */

