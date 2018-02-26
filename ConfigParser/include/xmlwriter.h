#ifndef XMLWRITER_H
#define XMLWRITER_H

#include <QFile>
#include <QXmlStreamWriter>
#include <list>
#include <map>
#include <string>
#include <ConfigParser/include/interface.h>

class XMLWriter
{
public:
    XMLWriter(QFile* filepath);
    void writeCaseConfig(CaseRegister* reg);
    void writeSystemConfig();

protected:
    QFile* file;
    QXmlStreamWriter xml;
};

#endif // XMLWRITER_H
