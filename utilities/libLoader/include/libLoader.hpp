#ifndef LIBLOADER_HPP
#define LIBLOADER_HPP

#include <string>

class libLoader
{
public:
    static void* open(const std::string& path);
    static void close(void* handler);
    template<typename T> static T resolve(void* handler, const std::string& func);
};

#endif // LIBLOADER_HPP
