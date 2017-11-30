#include "interface_visualization.h"
#include "Simplevisualizer.hpp"
#include <QApplication>
#include <iostream>

QApplication* app;

void* createVisualizer(int* argc, char** argv, void* Log)
{
    std::cout << "Entered createVisualizer\n";
    app = new QApplication(*argc, argv);
    std::cout << "Created QApplication\n";
    return (void*)(new SimpleVisualizer((logging::FileLogger*)Log));
}
