//===- TypeInference.cpp - Type inference passes -----------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "npcomp/Dialect/Basicpy/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::NPCOMP::Basicpy;

namespace {

class FunctionTypeInferencePass
    : public FunctionTypeInferenceBase<FunctionTypeInferencePass> {
  void runOnOperation() override {}
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::Basicpy::createFunctionTypeInferencePass() {
  return std::make_unique<FunctionTypeInferencePass>();
}
