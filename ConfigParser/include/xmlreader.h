#ifndef XMLREADER_H
#define XMLREADER_H

#include <QFile>
#include <QXmlStreamReader>
#include <list>
#include <map>
#include <string>
#include <ConfigParser/include/interface.h>

class XMLReader
{
public:
    XMLReader(QFile* filepath);
    // Plugin Reader
    std::string getPlugin(PluginType pluginType, std::string name);
    // Case config reader
    void readCaseConfig(CaseRegister* reg);

protected:
    // Plugin Reader
    enum PluginAutomataState { PASZeroState, PASCompScheme, PASGridModel };
    int findPlugin(std::string name, std::string* filename);
    PluginType getType(PluginAutomataState state);
    std::string getPluginPath(PluginType pluginType);
    // Case config reader
    enum CaseAutomataState { CASZeroState, CASMPI, CASSystem, CASSolver };
    void readSection(CaseAutomataState state, CaseRegister* reg);

protected:
    QFile* file;
    QXmlStreamReader xml;
};

#endif // XMLREADER_H
