#include <ComputationalScheme/include/interface.h>
#include <ComputationalScheme/include/LBM/LatticeBoltzmannScheme.hpp>
#include <ComputationalScheme/include/UnitTest/UnitTest_MovingBall.hpp>

void* createScheme(const char* schemeModel, const char* gridModel)
{
    //return (void*) (new LatticeBoltzmannScheme());
    return (void*) (new UnitTest_MovingBall());
}
