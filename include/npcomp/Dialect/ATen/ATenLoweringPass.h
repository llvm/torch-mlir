//===- ATenLoweringPass.h ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_ATEN_LOWERING_H
#define NPCOMP_DIALECT_ATEN_LOWERING_H

#include <memory>

namespace mlir {
class Pass;
class DialectConversion;
} // namespace mlir

namespace mlir {
namespace NPCOMP {
namespace aten {

std::unique_ptr<mlir::Pass> createATenLoweringPass();
void registerATenLoweringPass();
} // namespace aten
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_ATEN_LOWERING_H
