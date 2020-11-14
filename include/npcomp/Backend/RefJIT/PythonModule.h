//===- PythonModule.h - IREE python bindings ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_BACKEND_REFJIT_PYTHON_MODULE_H
#define NPCOMP_BACKEND_REFJIT_PYTHON_MODULE_H

#include "npcomp/Python/PybindUtils.h"

namespace npcomp {
namespace python {

/// Defines an "refjit" module with backend support definitions.
void defineBackendRefJitModule(py::module &m);

} // namespace python
} // namespace npcomp

#endif // NPCOMP_BACKEND_REFJIT_PYTHON_MODULE_H
