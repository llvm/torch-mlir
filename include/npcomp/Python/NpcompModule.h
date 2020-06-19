//===- NpcompModule.h - Module registrations ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_PYTHON_NPCOMP_MODULE_H
#define NPCOMP_PYTHON_NPCOMP_MODULE_H

#include "PybindUtils.h"

namespace mlir {
void defineMlirIrModule(py::module m);
void defineMlirPassModule(py::module m);
void defineMlirCoreDialects(py::module m);

namespace npcomp {
namespace python {

void defineNpcompDialect(py::module m);

} // namespace python
} // namespace npcomp
} // namespace mlir

#endif // NPCOMP_PYTHON_NPCOMP_MODULE_H
