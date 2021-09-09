//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/InitAll.h"

#include "iree-dialects/Dialect/IREE/IREEDialect.h"
#include "mlir/IR/Dialect.h"
#include "npcomp/Backend/Common/Passes.h"
#include "npcomp/Backend/IREE/Passes.h"
#include "npcomp/Conversion/Passes.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Basicpy/Transforms/Passes.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"
#include "npcomp/Dialect/Numpy/Transforms/Passes.h"
#include "npcomp/Dialect/Refback/IR/RefbackDialect.h"
#include "npcomp/Dialect/Refbackrt/IR/RefbackrtDialect.h"
#include "npcomp/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "npcomp/Dialect/TorchConversion/Transforms/Passes.h"
#include "npcomp/RefBackend/RefBackend.h"
#include "npcomp/Typing/Transforms/Passes.h"

void mlir::NPCOMP::registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<Basicpy::BasicpyDialect,
                  Numpy::NumpyDialect,
                  refbackrt::RefbackrtDialect,
                  refback::RefbackDialect,
                  mlir::NPCOMP::TorchConversion::TorchConversionDialect,
                  iree::IREEDialect>();
  // clang-format on
}

void mlir::NPCOMP::registerAllPasses() {
  mlir::NPCOMP::registerRefBackendPasses();
  mlir::NPCOMP::registerConversionPasses();
  mlir::NPCOMP::registerBasicpyPasses();
  mlir::NPCOMP::registerNumpyPasses();
  mlir::NPCOMP::registerTorchConversionPasses();
  mlir::NPCOMP::registerTypingPasses();
  mlir::NPCOMP::IREEBackend::registerIREEBackendPasses();
  mlir::NPCOMP::CommonBackend::registerCommonBackendPasses();
}
