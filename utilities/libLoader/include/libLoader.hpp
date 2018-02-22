#ifndef LIBLOADER_HPP
#define LIBLOADER_HPP

#include <string>
#include <dlfcn.h>

typedef void* DLHandler;

class libLoader
{
public:
    static DLHandler open(const std::string& path);
    static void close(void* handler);
    template<typename T> static T resolve(DLHandler handler, const std::string& func) {
        void* funcPtr = dlsym(handler, func.c_str());
        if(funcPtr == nullptr) {
            throw std::runtime_error("Can't load the function from the library!");
        }
        return reinterpret_cast<T>(funcPtr);
    }
};

#endif // LIBLOADER_HPP
