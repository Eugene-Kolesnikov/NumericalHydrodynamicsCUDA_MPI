#include "computationalScheme_interface.h"
#include "LatticeBoltzmannScheme.hpp"

void* createScheme(const char* schemeModel, const char* gridModel)
{
    return (void*) (new LatticeBoltzmannScheme());
}