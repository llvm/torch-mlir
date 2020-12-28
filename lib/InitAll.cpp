//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/InitAll.h"

#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "npcomp/Dialect/ATen/Transforms/Passes.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Basicpy/Transforms/Passes.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"
#include "npcomp/Dialect/Numpy/Transforms/Passes.h"
#include "npcomp/Dialect/RD/IR/RDDialect.h"
#include "npcomp/Dialect/RD/Transforms/Passes.h"
#include "npcomp/Dialect/Refback/IR/RefbackDialect.h"
#include "npcomp/Dialect/Refbackrt/IR/RefbackrtDialect.h"
#include "npcomp/Dialect/TCF/IR/TCFDialect.h"
#include "npcomp/Dialect/TCF/Transforms/Passes.h"
#include "npcomp/Dialect/TCP/IR/TCPDialect.h"
#include "npcomp/Dialect/TCP/Transforms/Passes.h"
#include "npcomp/Dialect/Torch/IR/TorchDialect.h"
#include "npcomp/Typing/Transforms/Passes.h"

#include "npcomp/Conversion/Passes.h"
#include "npcomp/RefBackend/RefBackend.h"

void mlir::NPCOMP::registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<mlir::NPCOMP::aten::ATenDialect,
                  Basicpy::BasicpyDialect,
                  Numpy::NumpyDialect,
                  mlir::NPCOMP::rd::RDDialect,
                  refbackrt::RefbackrtDialect,
                  refback::RefbackDialect,
                  tcf::TCFDialect,
                  tcp::TCPDialect,
                  mlir::NPCOMP::Torch::TorchDialect>();
  // clang-format on
}

void mlir::NPCOMP::registerAllPasses() {
  mlir::NPCOMP::registerATenPasses();
  mlir::NPCOMP::registerRefBackendPasses();
  mlir::NPCOMP::registerConversionPasses();
  mlir::NPCOMP::registerBasicpyPasses();
  mlir::NPCOMP::registerNumpyPasses();
  mlir::NPCOMP::registerRDPasses();
  mlir::NPCOMP::registerTCFPasses();
  mlir::NPCOMP::registerTCPPasses();
  mlir::NPCOMP::registerTypingPasses();
}
