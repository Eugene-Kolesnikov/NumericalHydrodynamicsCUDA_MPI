#include "computationalScheme_interface.h"
#include "LatticeBoltzmannScheme.hpp"
#include "UnitTest/UnitTest_MovingBall.hpp"

void* createScheme(const char* schemeModel, const char* gridModel)
{
    //return (void*) (new LatticeBoltzmannScheme());
    return (void*) (new UnitTest_MovingBall());
}
