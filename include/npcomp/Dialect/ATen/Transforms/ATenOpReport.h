//===- ATenOpReport.h -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_ATEN_OPREPORT_H
#define NPCOMP_DIALECT_ATEN_OPREPORT_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace mlir {
namespace NPCOMP {
namespace aten {

// Generate report on standard error.
std::unique_ptr<mlir::Pass> createATenOpReportPass();
// Return the report in the given output string.
std::unique_ptr<mlir::Pass> createATenOpReportPass(std::string &output);
void registerATenOpReportPass();

} // namespace aten
} // namespace NPCOMP
} // namespace mlir

#endif
