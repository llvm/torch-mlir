//===- MlirInit.h - MLIR config and init ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

// Note that the CPP module is compiled without RTTI or exceptions, unlike
// the rest of the pybind code. Therefore, we also stash some trampolines
// here for parts of the code that are not RTTI-compatible.

namespace mlir {

namespace npcomp {
namespace python {

// One time initialization.
bool npcompMlirInitialize();

} // namespace python
} // namespace npcomp
} // namespace mlir
