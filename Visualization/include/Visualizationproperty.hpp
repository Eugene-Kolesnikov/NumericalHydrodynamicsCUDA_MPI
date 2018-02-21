#ifndef VISUALIZATIONPROPERTY_HPP
#define VISUALIZATIONPROPERTY_HPP

#include <string>
#include <cstdlib>
#include <cstddef>
#include <typeindex>

struct VisualizationProperty
{
    std::string propertyName;
    size_t offset;
    size_t variables;
    std::type_index typeInfo;
    VisualizationProperty(const std::string& _propertyName,
        size_t _offset, size_t _variables,
        const std::type_index& _typeInfo): propertyName(_propertyName),
        offset(_offset), variables(_variables), typeInfo(_typeInfo) {}
};

#endif // VISUALIZATIONPROPERTY_HPP
