//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_CONVERSION_TCPTOLINALG_CONVERTTCPTOLINALG_H
#define NPCOMP_CONVERSION_TCPTOLINALG_CONVERTTCPTOLINALG_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace NPCOMP {
std::unique_ptr<OperationPass<ModuleOp>> createConvertTCPToLinalgPass();
}
}

#endif // NPCOMP_CONVERSION_TCPTOLINALG_CONVERTTCPTOLINALG_H
