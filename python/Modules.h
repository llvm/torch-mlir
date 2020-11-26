//===- Modules.h - Definitions for module creation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Python/PybindUtils.h"

namespace mlir {
namespace npcomp {
namespace python {

/// Populates bindings for the _npcomp.types module.
void populateTypesModule(py::module &m);

} // namespace python
} // namespace npcomp
} // namespace mlir
