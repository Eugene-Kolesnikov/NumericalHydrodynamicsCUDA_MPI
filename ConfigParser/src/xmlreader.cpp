#include <ConfigParser/include/xmlreader.h>
#include <cstdio>

XMLReader::XMLReader(QFile* filepath)
{
    file = filepath;
    xml.setDevice(file);
}

std::list<std::pair<std::string,double>>* XMLReader::readConfig()
{
    using namespace std;
    list<pair<string,double>>* params = nullptr;
    if (xml.readNextStartElement() && xml.name() == "xml" && xml.attributes().value("version") == "1.0") {
        params = new list<pair<string,double>>;
        while(!xml.atEnd() && !xml.hasError()) {
            auto token = xml.readNext();
            if(token == QXmlStreamReader::StartElement)
            {
                if(!(xml.name().toString() == "MPIConfiguration" || xml.name().toString() == "GPUConfiguration" ||
                     xml.name().toString() == "SolverConfiguration" || xml.name().toString() == "ComputationalModel" ||
                     xml.name().toString() == "GridModel" || xml.name().toString() == "Application")) {
                    params->push_back(make_pair<string,double>(xml.name().toString().toStdString(),xml.readElementText().toDouble()));
                }
            }
        }
    }
    return params;
}
