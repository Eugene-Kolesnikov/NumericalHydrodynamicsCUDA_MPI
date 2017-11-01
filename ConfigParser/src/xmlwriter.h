#ifndef XMLWRITER_H
#define XMLWRITER_H

#include <QFile>
#include <QXmlStreamWriter>
#include <list>
#include <map>
#include <string>

class XMLWriter
{
public:
    XMLWriter(QFile* filepath);
    void writeConfig(std::list<std::pair<std::string,double>>* params);

protected:
    QFile* file;
    QXmlStreamWriter xml;
};

#endif // XMLWRITER_H
