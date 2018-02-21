#include <ConfigParser/include/xmlwriter.h>

XMLWriter::XMLWriter(QFile* filepath)
{
    file = filepath;
    xml.setDevice(file);
    xml.setAutoFormatting(true);
}

void XMLWriter::writeConfig(std::list<std::pair<std::string,double>>* params)
{
    xml.writeStartDocument();
    xml.writeDTD("<!DOCTYPE xml>");
    xml.writeStartElement("xml");
    xml.writeAttribute("version", "1.0");
    for(auto it = params->begin(); it != params->end(); ++it) {
        if(it->first == std::string("MPI_NODES_X")) {
            xml.writeStartElement("MPIConfiguration");
        } else if(it->first == std::string("CUDA_X_THREADS")) {
            xml.writeStartElement("GPUConfiguration");
        } else if(it->first == std::string("TAU")) {
            xml.writeStartElement("SolverConfiguration");
        } else if(it->first == std::string("LBM")) {
            xml.writeStartElement("ComputationalModel");
        } else if(it->first == std::string("USG")) {
            xml.writeStartElement("GridModel");
        } else {
            // nothing
        }

        xml.writeTextElement(QString::fromStdString(it->first), QString::number(it->second));

        if(it->first == std::string("MPI_NODES_Y")) {
            xml.writeEndElement();
        } else if(it->first == std::string("CUDA_Y_THREADS")) {
            xml.writeEndElement();
        } else if(it->first == std::string("N_Y")) {
            xml.writeEndElement();
        } else if(it->first == std::string("NS")) {
            xml.writeEndElement();
        } else if(it->first == std::string("RND_TR")) {
            xml.writeEndElement();
        } else {
            // nothing
        }
    }
    xml.writeEndDocument();
}
