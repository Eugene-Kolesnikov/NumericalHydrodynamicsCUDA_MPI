#ifndef XMLREADER_H
#define XMLREADER_H

#include <QFile>
#include <QXmlStreamReader>
#include <list>
#include <map>
#include <string>

class XMLReader
{
public:
    XMLReader(QFile* filepath);
    std::list<std::pair<std::string,double>>* readConfig();

protected:
    QFile* file;
    QXmlStreamReader xml;
};

#endif // XMLREADER_H
