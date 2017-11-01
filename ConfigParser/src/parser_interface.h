#ifndef PARSER_INTERFACE_H
#define PARSER_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

void createConfig(void* list, const char* filepath);
void* readConfig(const char* filepath);

#ifdef __cplusplus
}
#endif

#endif // PARSER_INTERFACE_H
