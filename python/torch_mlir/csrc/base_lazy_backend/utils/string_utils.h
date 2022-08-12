#pragma once

#include <string>
#include <sstream>
#include <vector>


template <typename T>
std::ostream& string_join(std::ostream& out, const std::vector<T>& v, const std::string& delimiter) {
    size_t i = 0;
    for (const T& e : v) {
        if ((i++) > 0) { out << delimiter; }
        out << e;
    }
    return out;
}

template <typename T>
std::string string_join(const std::vector<T>& v, const std::string& delimiter) {
    std::ostringstream joined;
    string_join(joined, v, delimiter);
    return joined.str();
}


/*
 * Returns true if str starts with prefix
 */
inline bool startswith(const std::string& str, const std::string& prefix) {
   return str.rfind(prefix, 0) == 0;
}
