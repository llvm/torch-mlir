//===- PybindUtils.cpp - Utilities for interop with python ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Python/PybindUtils.h"

namespace pybind11 {

pybind11::error_already_set raisePyError(PyObject *exc_class,
                                         const char *message) {
  PyErr_SetString(exc_class, message);
  return pybind11::error_already_set();
}

} // namespace pybind11
