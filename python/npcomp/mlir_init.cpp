//===- mlir_init.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllDialects.h"

namespace npcomp {
namespace python {

void npcompMlirInitialize() {
  ::mlir::registerAllDialects();
}

} // namespace python
} // namesapce npcomp
