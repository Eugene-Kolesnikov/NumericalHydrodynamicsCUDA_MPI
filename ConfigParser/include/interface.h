#ifndef PARSER_INTERFACE_H
#define PARSER_INTERFACE_H

#include <utilities/Register/include/Plugin.hpp>
#include <utilities/Register/include/CaseRegister.hpp>

#ifdef __cplusplus
extern "C" {
#endif

enum PluginType { SchemePlugin, GridPlugin };
void findPlugin(PluginType type, Plugin* plugin, std::string filepath);
void readCaseConfig(CaseRegister* reg, std::string filepath);
void createCaseConfig(CaseRegister* reg, const char* filepath);
void createSystemConfig(std::string filepath);

#ifdef __cplusplus
}
#endif

#endif // PARSER_INTERFACE_H
