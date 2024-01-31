//===- exception.h --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <exception>
#include <sstream>
#include <string>

#define UNIMPLEMENTED_ERROR(msg)                                               \
  {                                                                            \
    std::ostringstream err;                                                    \
    err << "Unimplemented Error: " << msg;                                     \
    throw std::runtime_error(err.str());                                       \
  }

#define UNIMPLEMENTED_FUNCTION_ERROR()                                         \
  UNIMPLEMENTED_ERROR("\n\t" << __FILE__ << ":" << __LINE__ << "    "          \
                             << __PRETTY_FUNCTION__)

#define UNSUPPORTED_ERROR(msg)                                                 \
  {                                                                            \
    std::ostringstream err;                                                    \
    err << "Unsupported Error: " << msg;                                       \
    throw std::runtime_error(err.str());                                       \
  }
