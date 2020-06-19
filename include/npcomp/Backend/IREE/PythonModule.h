//===- PythonModule.h - IREE python bindings ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_BACKEND_IREE_PYTHON_MODULE_H
#define NPCOMP_BACKEND_IREE_PYTHON_MODULE_H

#include "npcomp/Python/PybindUtils.h"

namespace mlir {
namespace npcomp {
namespace python {

/// Defines an "iree" module with backend support definitions.
void defineBackendIREEModule(py::module m);

} // namespace python
} // namespace npcomp
} // namespace mlir

#endif // NPCOMP_BACKEND_IREE_PYTHON_MODULE_H
