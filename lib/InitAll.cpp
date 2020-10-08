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
#include "npcomp/Dialect/Refbackrt/IR/RefbackrtDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"
#include "npcomp/Dialect/Numpy/Transforms/Passes.h"
#include "npcomp/Dialect/RefBackend/IR/RefBackendDialect.h"
#include "npcomp/Dialect/TCF/IR/TCFDialect.h"
#include "npcomp/Dialect/TCF/Transforms/Passes.h"
#include "npcomp/Dialect/TCP/IR/TCPDialect.h"
#include "npcomp/Dialect/Torch/IR/TorchDialect.h"
#include "npcomp/Typing/Transforms/Passes.h"

#include "npcomp/Conversion/Passes.h"
#include "npcomp/RefBackend/RefBackend.h"

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

static void registerDependencyDialects(mlir::DialectRegistry &registry) {
#ifdef NPCOMP_ENABLE_IREE
  // TODO: We should probably be registering the MLIR dialects regardless
  // of building with IREE, but we have to do it with IREE, and the
  // dependencies are coming from there and wouldn't be great to duplicate.
  // See iree/tools:init_mlir_passes_and_dialects
  mlir::registerMlirDialects(registry);
  mlir::registerXLADialects(registry);
  mlir::iree_compiler::registerIreeDialects(registry);
  mlir::iree_compiler::registerIreeCompilerModuleDialects(registry);
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

void mlir::NPCOMP::registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<mlir::NPCOMP::aten::ATenDialect,
                  Basicpy::BasicpyDialect,
                  Numpy::NumpyDialect,
                  refbackrt::RefbackrtDialect,
                  refback::RefBackendDialect,
                  tcf::TCFDialect,
                  tcp::TCPDialect,
                  mlir::NPCOMP::Torch::TorchDialect>();
  // clang-format on
  registerDependencyDialects(registry);
}

void mlir::NPCOMP::registerAllPasses() {
  mlir::NPCOMP::aten::registerATenPasses();
  mlir::NPCOMP::registerRefBackendPasses();
  mlir::NPCOMP::registerConversionPasses();
  mlir::NPCOMP::registerBasicpyPasses();
  mlir::NPCOMP::registerNumpyPasses();
  mlir::NPCOMP::registerTCFPasses();
  mlir::NPCOMP::registerTypingPasses();
  registerDependencyPasses();
}
