#include <utilities/libLoader/include/libLoader.hpp>

void* libLoader::open(const std::string &path)
{
    void* handle = dlopen(path.c_str(), RTLD_NOW);
    if (handle == nullptr)
        throw std::runtime_error(dlerror());
    return handle;
}

void libLoader::close(void* handler)
{
    if(dlclose(handler))
        throw std::runtime_error("Dynamic library was not closed properly!");
}

template<typename T> T libLoader::resolve(void* handler, const std::string& func)
{
    void* funcPtr = dlsym(handler, func.c_str());
    if(funcPtr == nullptr) {
        throw std::runtime_error("Can't load the function from the Computational scheme library!");
    }
    return funcPtr;
}

/** Example of using:
int main() {
    void* libHandler;
    std::string lib = ".../libTest.so";
    std::string func_name = "createTestClass";
    libHandler = libLoader::open(lib);
    auto createClass = libLoader::resolve<decltype(&createTestClass)>(libHandler, func_name);
    base* scheme = (base*)createClass();
    std::cout << scheme->compute(10) << std::endl;
    delete scheme;
    libLoader::close(libHandler);
    return 0;
}
 */
