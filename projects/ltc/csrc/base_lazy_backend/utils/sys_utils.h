#pragma once

#include <cstdlib>
#include <cstring>

namespace sys_util {

template <typename T>
static T GetEnv(const std::string &name, const T &default_value = T(0)) {
  const char *env = std::getenv(name.c_str());
  if (!env) {
    return default_value;
  }
  return T(std::atoi(env));
}

[[maybe_unused]] static std::string
GetEnvString(const std::string &name, const std::string &default_value) {
  const char *env = std::getenv(name.c_str());
  if (!env) {
    return default_value;
  }
  return std::string(env);
}

[[maybe_unused]] static bool GetEnvBool(const char *name, bool defval) {
  const char *env = std::getenv(name);
  if (env == nullptr) {
    return defval;
  }
  if (std::strcmp(env, "true") == 0) {
    return true;
  }
  if (std::strcmp(env, "false") == 0) {
    return false;
  }
  return std::atoi(env) != 0;
}

} // namespace sys_util
