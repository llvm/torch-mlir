#pragma once

#include <sstream>
#include <string>
#include <vector>

template <typename T>
std::ostream &string_join(std::ostream &out, const std::vector<T> &v,
                          const std::string &delimiter) {
  size_t i = 0;
  for (const T &e : v) {
    if ((i++) > 0) {
      out << delimiter;
    }
    out << e;
  }
  return out;
}

template <typename T>
std::string string_join(const std::vector<T> &v, const std::string &delimiter) {
  std::ostringstream joined;
  string_join(joined, v, delimiter);
  return joined.str();
}

inline std::vector<std::string> string_split(const std::string &str,
                                             const std::string &sep) {
  std::vector<std::string> tokens;
  std::size_t pos1 = str.find_first_not_of(sep);
  while (pos1 != std::string::npos) {
    std::size_t pos2 = str.find_first_of(sep, pos1);
    if (pos2 == std::string::npos) {
      tokens.push_back(str.substr(pos1));
      pos1 = pos2;
    } else {
      tokens.push_back(str.substr(pos1, pos2 - pos1));
      pos1 = str.find_first_not_of(sep, pos2 + 1);
    }
  }
  return tokens;
}

/*
 * Returns true if str starts with prefix
 */
inline bool startswith(const std::string &str, const std::string &prefix) {
  return str.rfind(prefix, 0) == 0;
}
