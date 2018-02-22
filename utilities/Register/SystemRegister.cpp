#include <utilities/Register/SystemRegister.hpp>

std::string SystemRegister::ConfigParser::name = "libConfigParser.1.0.0.dylib"; // hardcoded
std::string SystemRegister::ConfigParser::interface[2] = {"createConfig", "readConfig"}; // hardcoded
std::string SystemRegister::ConfigFile;
std::string SystemRegister::VisLib::name;
std::string SystemRegister::VisLib::interface;
std::string SystemRegister::CompModel::name;
std::string SystemRegister::CompModel::interface;
std::string SystemRegister::CompScheme::name;
std::string SystemRegister::CompScheme::interface;
