#include "computationalScheme_interface.h"
#include "TestScheme.hpp"

void* createScheme(const char* schemeModel, const char* gridModel)
{
    return (void*) (new TestScheme());
}