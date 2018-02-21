#ifndef COMPUTATIONALSCHEME_INTERFACE_H
#define COMPUTATIONALSCHEME_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

void* createScheme(const char* schemeModel, const char* gridModel);

#ifdef __cplusplus
}
#endif

#endif /* COMPUTATIONALSCHEME_INTERFACE_H */