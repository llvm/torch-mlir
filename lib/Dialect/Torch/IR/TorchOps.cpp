//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Torch/IR/TorchOps.h"

#include "mlir/IR/Builders.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

#define GET_OP_CLASSES
#include "npcomp/Dialect/Torch/IR/TorchOps.cpp.inc"

static SmallVector<StringRef, 4> strArrayAttrToVector(ArrayAttr array) {
  SmallVector<StringRef, 4> strings;
  strings.reserve(array.size());
  for (auto stringAttr : array) {
    strings.push_back(stringAttr.cast<StringAttr>().getValue());
  }
  return strings;
}

// -----------------------------------------------------------------------------
// KernelCall op
// -----------------------------------------------------------------------------

Torch::KernelMetadata Torch::KernelCallOp::getTorchKernelMetadata() {
  return Torch::KernelMetadata{
      .kernelName = kernelName(),
      .isVararg = sigIsVararg(),
      .isVarret = sigIsVarret(),
      .argTypes = strArrayAttrToVector(sigArgTypes()),
      .returnTypes = strArrayAttrToVector(sigRetTypes()),
  };
}
