#include <ConfigParser/include/interface.h>
#include <ConfigParser/include/xmlreader.h>
#include <ConfigParser/include/xmlwriter.h>
#include <QFile>
#include <list>
#include <map>
#include <string>
#include <exception>
#include <iostream>


void findPlugin(PluginType type, Plugin* plugin, std::string filepath)
{
    QFile file(filepath.c_str());
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        throw std::runtime_error(file.errorString().toStdString());
    }
    XMLReader xmlreader(&file);
    plugin->DL_Location = xmlreader.getPlugin(type, plugin->Name);
    file.close();
}

void readCaseConfig(CaseRegister* reg, std::string filepath)
{
    QFile file(filepath.c_str());
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        throw std::runtime_error(file.errorString().toStdString());
    }
    XMLReader xmlreader(&file);
    xmlreader.readCaseConfig(reg);
    file.close();
}

void createCaseConfig(CaseRegister* reg, const char* filepath)
{
    QFile file(filepath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        throw std::runtime_error(file.errorString().toStdString());
    }
    XMLWriter xmlwriter(&file);
    xmlwriter.writeCaseConfig(reg);
    file.close();
}

void createSystemConfig(std::string filepath)
{
    QFile file(filepath.c_str());
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        throw std::runtime_error(file.errorString().toStdString());
    }
    XMLWriter xmlwriter(&file);
    xmlwriter.writeSystemConfig();
    file.close();
}
