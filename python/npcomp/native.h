//===- dialect.h - Module registrations -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_PYTHON_NATIVE_H
#define NPCOMP_PYTHON_NATIVE_H

#include "pybind_utils.h"

namespace mlir {
void defineMlirIrModule(py::module m);

namespace npcomp {
namespace python {

bool npcompMlirInitialize();
void defineNpcompDialect(py::module m);

} // namespace python
} // namespace npcomp
} // namespace mlir

#endif // NPCOMP_PYTHON_NATIVE_H
