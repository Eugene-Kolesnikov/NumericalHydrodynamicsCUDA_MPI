#ifndef SYSTEM_REGISTER_HPP
#define SYSTEM_REGISTER_HPP

#include <string>

struct SystemRegister
{
    struct VisLib {
        static std::string name;
        static std::string interface;
    };
    struct CompModel {
        static std::string name;
        static std::string interface;
    };
    struct CompScheme {
        static std::string name;
        static std::string interface;
    };
    struct ConfigParser {
        enum interfaceFunctions {createConfig = 0, readConfig = 1};
        static std::string name;
        static std::string interface[2];
    };
    static std::string ConfigFile;
};

#endif
