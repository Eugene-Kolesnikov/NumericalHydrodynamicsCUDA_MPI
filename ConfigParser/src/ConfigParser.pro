TARGET = ConfigParser

QT += xml

TEMPLATE = lib

DEFINES += CONFIG_PARSER_LIB

DESTDIR = ../build
OBJECTS_DIR = $$DESTDIR
MOC_DIR = $$DESTDIR
RCC_DIR = $$DESTDIR
UI_DIR = $$DESTDIR

SOURCES += main.cpp \
    parser_interface.cpp \
    xmlreader.cpp \
    xmlwriter.cpp

HEADERS += \
    parser_interface.h \
    xmlreader.h \
    xmlwriter.h
