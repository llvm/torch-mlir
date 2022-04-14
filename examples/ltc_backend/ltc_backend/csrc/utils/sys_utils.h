//===- sys_utils.h --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdlib>
#include <string>

namespace sys_util {

template <typename T>
T GetEnv(const std::string &name, const T &default_value = T(0)) {
  const char *env = std::getenv(name.c_str());
  if (!env) {
    return default_value;
  }
  return T(std::atoi(env));
}

} // namespace sys_util
