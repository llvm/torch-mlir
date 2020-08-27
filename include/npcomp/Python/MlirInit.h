//===- MlirInit.h - MLIR config and init ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_PYTHON_MLIRINIT_H
#define NPCOMP_PYTHON_MLIRINIT_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

// Note that the CPP module is compiled without RTTI or exceptions, unlike
// the rest of the pybind code. Therefore, we also stash some trampolines
// here for parts of the code that are not RTTI-compatible.

namespace mlir {

class MLIRContext;

namespace npcomp {
namespace python {

// One time initialization.
bool npcompMlirInitialize();

// Loads globally registered dialects into the MLIRContext.
// This is temporary until there is an upstream story for handling dialect
// registration in python-based systems.
void loadGlobalDialectsIntoContext(MLIRContext *context);

} // namespace python
} // namespace npcomp
} // namespace mlir

#endif // NPCOMP_PYTHON_MLIRINIT_H
