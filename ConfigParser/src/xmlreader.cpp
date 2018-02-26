#include <ConfigParser/include/xmlreader.h>
#include <cstdio>
#include <iostream>

#ifdef Q_OS_LINUX
    #define DLEXT ".so"
#endif
#ifdef Q_OS_MACOS
    #define DLEXT ".dylib"
#endif

XMLReader::XMLReader(QFile* filepath)
{
    file = filepath;
    xml.setDevice(file);
}

std::string XMLReader::getPlugin(PluginType pluginType, std::string name)
{
    PluginAutomataState state = PASZeroState;
    using namespace std;
    string tmp;
    string filename;
    int error;
    if (xml.readNextStartElement() && xml.name() == "xml" && xml.attributes().value("version") == "1.0") {
        while(!xml.atEnd() && !xml.hasError()) {
            auto token = xml.readNext();
            tmp = xml.name().toString().toStdString();
            if(token == QXmlStreamReader::StartElement)
            {
                if(tmp == "ComputationalScheme") {
                    state = PASCompScheme;
                    continue;
                } else if(tmp == "GridModel") {
                    state = PASGridModel;
                    continue;
                } else {
                    if(getType(state) == pluginType) {
                        error = findPlugin(name, &filename);
                        break;
                    }
                }
            }
            else if(token == QXmlStreamReader::EndElement)
            {
                if(state == PASCompScheme && tmp == "ComputationalScheme") {
                    state = PASZeroState;
                } else if(state == PASGridModel && tmp == "GridModel") {
                    state = PASZeroState;
                }
            }
        }
        if(error == 0) {
            return (getPluginPath(pluginType) + filename + DLEXT);
        } else {
           throw std::runtime_error("Couldn't find the requested plugin in the System Config File!");
        }
    } else {
        throw runtime_error("Unknown file format!");
    }
}

int XMLReader::findPlugin(std::string name, std::string* filename)
{
    using namespace std;
    string tmp;
    int level = 0;
    while(!xml.atEnd() && !xml.hasError() && level >= 0) {
        auto token = xml.readNext();
        tmp = xml.name().toString().toStdString();
        if(token == QXmlStreamReader::StartElement)
        {
            ++level;
            if(tmp == name) {
                *filename = xml.readElementText().toStdString();
                return 0;
            }
        }
        else if(token == QXmlStreamReader::EndElement)
        {
            --level;
        }
    }
    return 1;
}

PluginType XMLReader::getType(PluginAutomataState state)
{
    if(state == PASCompScheme)
        return SchemePlugin;
    else if(state == PASGridModel)
        return GridPlugin;
}

void XMLReader::readCaseConfig(CaseRegister* reg)
{
    CaseAutomataState state = CASZeroState;
    using namespace std;
    string tmp;
    if (xml.readNextStartElement() && xml.name() == "xml" && xml.attributes().value("version") == "1.0") {
        while(!xml.atEnd() && !xml.hasError()) {
            auto token = xml.readNext();
            tmp = xml.name().toString().toStdString();
            if(token == QXmlStreamReader::StartElement)
            {
                if(tmp == "MPIParameters") {
                    state = CASMPI;
                    continue;
                } else if(tmp == "SystemParameters") {
                    state = CASSystem;
                    continue;
                } else if(tmp == "SolverParameters") {
                    state = CASSolver;
                    continue;
                }
                readSection(state, reg);
            }
        }
    }
}

void XMLReader::readSection(CaseAutomataState state, CaseRegister* reg)
{
    using namespace std;
    string tmp;
    int level = 0;
    while(!xml.atEnd() && !xml.hasError() && level >= 0) {
        auto token = xml.readNext();
        tmp = xml.name().toString().toStdString();
        if(token == QXmlStreamReader::StartElement)
        {
            ++level;
            if(state == CASMPI) {
                if(tmp == "MPI_NODES_X") {
                    reg->MPI_NODES_X = xml.readElementText().toUInt();
                } else if(tmp == "MPI_NODES_Y") {
                    reg->MPI_NODES_Y = xml.readElementText().toUInt();
                }
            } else if(state == CASSystem) {
                if(tmp == "N_X") {
                    reg->N_X = xml.readElementText().toUInt();
                } else if(tmp == "N_Y") {
                    reg->N_Y = xml.readElementText().toUInt();
                } else if(tmp == "X_MAX") {
                    reg->X_MAX = xml.readElementText().toDouble();
                } else if(tmp == "Y_MAX") {
                    reg->Y_MAX = xml.readElementText().toDouble();
                } else if(tmp == "TAU") {
                    reg->TAU = xml.readElementText().toDouble();
                } else if(tmp == "TOTAL_TIME") {
                    reg->TOTAL_TIME = xml.readElementText().toDouble();
                } else if(tmp == "STEP_LENGTH") {
                    reg->STEP_LENGTH = xml.readElementText().toUInt();
                }
            } else if(state == CASSolver) {
                if(tmp == "Scheme") {
                    reg->scheme = xml.readElementText().toStdString();
                } else if(tmp == "Grid") {
                    reg->grid = xml.readElementText().toStdString();
                }
            }
        }
        else if(token == QXmlStreamReader::EndElement)
        {
            --level;
        }
    }
}

std::string XMLReader::getPluginPath(PluginType pluginType)
{
    if(pluginType == SchemePlugin) {
        return std::string("lib/scheme/");
    } else if(pluginType == GridPlugin) {
        return std::string("lib/grid/");
    }
    throw std::runtime_error("Unknown Plugin Type!");
}
