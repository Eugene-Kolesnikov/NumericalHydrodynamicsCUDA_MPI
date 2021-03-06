#-------------------------------------------------
#
# Project created by QtCreator 2017-11-24T16:31:56
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets  printsupport

TARGET = Visualization

DEFINES += VISUALIZATION_LIB

INCLUDEPATH += ./../../

TEMPLATE = lib
VERSION = 2.0.0

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

DESTDIR = ../build
OBJECTS_DIR = $$DESTDIR
MOC_DIR = $$DESTDIR
RCC_DIR = $$DESTDIR
UI_DIR = $$DESTDIR

SOURCES += \
        interface.cpp \
        Mainwindow.cpp \
        ../../utilities/QCustomPlot/src/Qcustomplot.cpp \
        Simplevisualizer.cpp \
        Visualizer.cpp \
        ../../utilities/Logger/src/FileLogger.cpp

HEADERS += \
        ../include/Mainwindow.hpp \
        ../include/Simplevisualizer.hpp \
        ../include/Visualizer.hpp \
        ../../utilities/QCustomPlot/include/Qcustomplot.hpp \
        ../include/interface.h \
        ../../utilities/Logger/include/FileLogger.hpp \
        ../include/Visualizationproperty.hpp

FORMS += \
        mainwindow.ui
