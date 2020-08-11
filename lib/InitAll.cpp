//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/InitAll.h"

#include "npcomp/Dialect/ATen/ATenDialect.h"
#include "npcomp/Dialect/ATen/ATenPasses.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Basicpy/Transforms/Passes.h"
#include "npcomp/Dialect/Npcomprt/IR/NpcomprtDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"
#include "npcomp/Dialect/Numpy/Transforms/Passes.h"
#include "npcomp/Dialect/TCF/IR/TCFDialect.h"
#include "npcomp/Dialect/TCF/Transforms/Passes.h"
#include "npcomp/Dialect/TCP/IR/TCPDialect.h"
#include "npcomp/Typing/Transforms/Passes.h"

#include "npcomp/Conversion/BasicpyToStd/Passes.h"
#include "npcomp/Conversion/NumpyToTCF/Passes.h"
#include "npcomp/Conversion/TCFToTCP/TCFToTCP.h"
#include "npcomp/Conversion/TCPToLinalg/TCPToLinalg.h"
#include "npcomp/E2E/E2E.h"

#ifdef NPCOMP_ENABLE_IREE
#include "iree/tools/init_compiler_modules.h"
#include "iree/tools/init_iree_dialects.h"
#include "iree/tools/init_iree_passes.h"
#include "iree/tools/init_mlir_dialects.h"
#include "iree/tools/init_mlir_passes.h"
#include "iree/tools/init_targets.h"
#include "iree/tools/init_xla_dialects.h"
// TODO: For some reason these aren't bundled with the rest.
#include "iree/compiler/Conversion/HLOToLinalg/Passes.h"
#include "iree/compiler/Conversion/init_conversions.h"
#include "iree/compiler/Dialect/HAL/Conversion/Passes.h"
#endif // NPCOMP_ENABLE_IREE

static void registerDependencyDialects() {
#ifdef NPCOMP_ENABLE_IREE
  // TODO: We should probably be registering the MLIR dialects regardless
  // of building with IREE, but we have to do it with IREE, and the
  // dependencies are coming from there and wouldn't be great to duplicate.
  // See iree/tools:init_mlir_passes_and_dialects
  mlir::registerMlirDialects();
  mlir::registerXLADialects();
  mlir::iree_compiler::registerIreeDialects();
  mlir::iree_compiler::registerIreeCompilerModuleDialects();
#endif // NPCOMP_ENABLE_IREE
}

static void registerDependencyPasses() {
#ifdef NPCOMP_ENABLE_IREE
  // TODO: We should probably be registering the MLIR passes regardless
  // of building with IREE, but we have to do it with IREE, and the
  // dependencies are coming from there and wouldn't be great to duplicate.
  // See iree/tools:init_mlir_passes_and_dialects
  mlir::registerMlirPasses();
  mlir::iree_compiler::registerAllIreePasses();
  mlir::iree_compiler::registerHALConversionPasses();
  mlir::iree_compiler::registerHALTargetBackends();
  mlir::iree_compiler::registerLinalgToSPIRVPasses();
  mlir::iree_compiler::registerHLOToLinalgPasses();
  mlir::iree_compiler::registerLinalgToLLVMPasses();
#endif // NPCOMP_ENABLE_IREE
}

void mlir::NPCOMP::registerAllDialects() {
  registerDialect<mlir::NPCOMP::aten::ATenDialect>();
  registerDialect<Basicpy::BasicpyDialect>();
  registerDialect<Numpy::NumpyDialect>();
  registerDialect<npcomprt::NpcomprtDialect>();
  registerDialect<tcf::TCFDialect>();
  registerDialect<tcp::TCPDialect>();
  registerDependencyDialects();
}

void mlir::NPCOMP::registerAllPasses() {
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
#define GEN_PASS_REGISTRATION
#include "npcomp/Dialect/Basicpy/Transforms/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "npcomp/Dialect/Numpy/Transforms/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "npcomp/Dialect/TCF/Transforms/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "npcomp/Typing/Transforms/Passes.h.inc"
  mlir::NPCOMP::aten::registerATenPasses();
  registerDependencyPasses();
}
