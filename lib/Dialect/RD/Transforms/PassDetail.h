//===- PassDetail.h - Pass details ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_RD_TRANSFORMS_PASSDETAIL_H
#define NPCOMP_DIALECT_RD_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"
#include "npcomp/Dialect/RD/IR/RDOps.h"

namespace mlir {
namespace NPCOMP {

#define GEN_PASS_CLASSES
#include "npcomp/Dialect/RD/Transforms/Passes.h.inc"

} // namespace NPCOMP
} // end namespace mlir

namespace mlir {
namespace NPCOMP {
// Returns the FuncOp that definesthe pipeline.
llvm::Optional<FuncOp> findDefinitionFunc(rd::PipelineDefinitionOp definition);
}
}

#endif // NPCOMP_DIALECT_RD_TRANSFORMS_PASSDETAIL_H
