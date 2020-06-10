//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_INITALL_H
#define NPCOMP_INITALL_H

#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/NpcompRt/IR/NpcompRtDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"
#include "npcomp/Dialect/TCF/IR/TCFDialect.h"
#include "npcomp/Dialect/TCP/IR/TCPDialect.h"

#include "npcomp/Conversion/TCFToTCP/TCFToTCP.h"
#include "npcomp/Conversion/TCPToLinalg/TCPToLinalg.h"
#include "npcomp/E2E/E2E.h"


namespace mlir {
namespace NPCOMP {

inline void registerAllDialects() {
  registerDialect<Basicpy::BasicpyDialect>();
  registerDialect<Numpy::NumpyDialect>();
  registerDialect<npcomp_rt::NpcompRtDialect>();
  registerDialect<tcf::TCFDialect>();
  registerDialect<tcp::TCPDialect>();
}

inline void registerAllPasses() {
  using mlir::Pass; // The .inc files reference this unqualified.
#define GEN_PASS_REGISTRATION
#include "npcomp/E2E/Passes.h.inc"
  // TODO: The following pipeline registration uses pass manager options,
  // which causes vague linkage issues when co-mingled with code that
  // uses RTTI.
  mlir::PassPipelineRegistration<E2ELoweringPipelineOptions>(
      "e2e-lowering-pipeline", "E2E lowering pipeline.",
      mlir::NPCOMP::createE2ELoweringPipeline);
  mlir::PassPipelineRegistration<>(
      "lower-to-hybrid-tensor-memref-pipeline",
      "Pipeline lowering to hybrid tensor/memref.",
      mlir::NPCOMP::createLowerToHybridTensorMemRefPipeline);
#define GEN_PASS_REGISTRATION
#include "npcomp/Conversion/Passes.h.inc"
}
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_INITALL_H
