//===- MlirPass.h - MLIR Pass Bindings ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_PYTHON_MLIR_PASS_H
#define NPCOMP_PYTHON_MLIR_PASS_H

#include "MlirIr.h"
#include "PybindUtils.h"

#include "mlir/Pass/PassManager.h"

namespace mlir {

struct PyPassManager {
  PyPassManager(std::shared_ptr<PyContext> context, bool verifyModules)
      : passManager(&context->context, OpPassManager::Nesting::Implicit),
        context(std::move(context)) {
    passManager.enableVerifier(verifyModules);
  }
  static void bind(py::module m);
  PassManager passManager;

private:
  std::shared_ptr<PyContext> context;
};

} // namespace mlir

#endif // NPCOMP_PYTHON_MLIR_PASS_H
