//===- debug.h ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <iostream>

#include "sys_utils.h"

#define PRINT_DEBUG(msg)                                                       \
  std::cout << msg << "    (" << __FILE__ << ":" << __LINE__ << ")"            \
            << std::endl;

#define PRINT_FUNCTION()                                                       \
  if (verbose_print_function) {                                                \
    std::cout << __PRETTY_FUNCTION__ << "    (" << __FILE__ << ":" << __LINE__ \
              << ")" << std::endl;                                             \
  }

static const bool verbose_print_function =
    sys_util::GetEnvBool("VERBOSE_PRINT_FUNCTION", false);
