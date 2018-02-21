TARGET = ConfigParser

QT += xml

TEMPLATE = lib

INCLUDEPATH += ./../../

DEFINES += CONFIG_PARSER_LIB

DESTDIR = ../build
OBJECTS_DIR = $$DESTDIR
MOC_DIR = $$DESTDIR
RCC_DIR = $$DESTDIR
UI_DIR = $$DESTDIR

SOURCES += \
    interface.cpp \
    xmlreader.cpp \
    xmlwriter.cpp

HEADERS += \
    ../include/interface.h \
    ../include/xmlreader.h \
    ../include/xmlwriter.h
