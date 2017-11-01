TARGET = ConfigParser

QT += xml

TEMPLATE = lib

DEFINES += CONFIG_PARSER_LIB

SOURCES += main.cpp \
    parser_interface.cpp \
    xmlreader.cpp \
    xmlwriter.cpp

HEADERS += \
    parser_interface.h \
    xmlreader.h \
    xmlwriter.h
