#include "parser_interface.h"
#include "xmlreader.h"
#include "xmlwriter.h"
#include <QFile>
#include <list>
#include <map>
#include <string>
#include <exception>

void createConfig(void* list, const char* filepath)
{
    QFile file(filepath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        throw std::runtime_error(file.errorString().toStdString());
        return;
    }
    std::list<std::pair<std::string,double>>* params =
            (std::list<std::pair<std::string,double>>*) list;
    XMLWriter xmlwriter(&file);
    xmlwriter.writeConfig(params);
    file.close();
}

void* readConfig(const char* filepath)
{
    QFile file(filepath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        throw std::runtime_error(file.errorString().toStdString());
        return nullptr;
    }
    XMLReader xmlreader(&file);
    std::list<std::pair<std::string,double>>* params = xmlreader.readConfig();
    file.close();
    return (void*)params;
}
