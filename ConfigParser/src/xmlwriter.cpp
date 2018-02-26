#include <ConfigParser/include/xmlwriter.h>

using namespace std;

XMLWriter::XMLWriter(QFile* filepath)
{
    file = filepath;
    xml.setDevice(file);
    xml.setAutoFormatting(true);
}

void XMLWriter::writeCaseConfig(CaseRegister* reg)
{
    xml.writeStartDocument();
    xml.writeDTD("<!DOCTYPE xml>");
    xml.writeStartElement("xml");
    xml.writeAttribute("version", "1.0");
    xml.writeStartElement("MPIParameters");
    xml.writeTextElement("MPI_NODES_X",QString::number(reg->MPI_NODES_X));
    xml.writeTextElement("MPI_NODES_Y",QString::number(reg->MPI_NODES_Y));
    xml.writeEndElement();
    xml.writeStartElement("SystemParameters");
    xml.writeTextElement("N_X",QString::number(reg->N_X));
    xml.writeTextElement("N_Y",QString::number(reg->N_Y));
    xml.writeTextElement("X_MAX",QString::number(reg->X_MAX));
    xml.writeTextElement("Y_MAX",QString::number(reg->Y_MAX));
    xml.writeTextElement("TAU",QString::number(reg->TAU));
    xml.writeTextElement("TOTAL_TIME",QString::number(reg->TOTAL_TIME));
    xml.writeTextElement("STEP_LENGTH",QString::number(reg->STEP_LENGTH));
    xml.writeEndElement();
    xml.writeStartElement("SolverParameters");
    xml.writeTextElement("Scheme",QString::fromStdString(reg->scheme));
    xml.writeTextElement("Grid",QString::fromStdString(reg->grid));
    xml.writeEndElement();
    xml.writeEndDocument();
}

void XMLWriter::writeSystemConfig()
{
    xml.writeStartDocument();
    xml.writeDTD("<!DOCTYPE xml>");
    xml.writeStartElement("xml");
    xml.writeAttribute("version", "1.0");
    xml.writeStartElement("ComputationalScheme");
    xml.writeTextElement("Lattice Boltzmann Method","libLBM");
    xml.writeTextElement("Finite Volume Method","libFVM");
    xml.writeTextElement("Marker in Cell Method","libMIC");
    xml.writeEndElement();
    xml.writeStartElement("GridModel");
    xml.writeTextElement("Uniform Grid","libUniform");
    xml.writeEndElement();
    xml.writeEndDocument();
}
