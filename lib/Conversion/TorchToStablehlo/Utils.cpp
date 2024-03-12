//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "./Utils.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace torch;

FailureOr<Type> torch_to_stablehlo::getBackendTypeForScalarType(
    MLIRContext *context, torch_upstream::ScalarType dtypeInt) {
  FailureOr<Type> maybeType = Torch::getTypeForScalarType(
      context, (torch_upstream::ScalarType)dtypeInt);
  if (failed(maybeType)) {
    return failure();
  }
  Type type = *maybeType;
  // The stablehlo backend expects signed integers to be signless.
  if (type.isSignedInteger()) {
    type = IntegerType::get(context, type.getIntOrFloatBitWidth(),
                            IntegerType::Signless);
  }
  return type;
}
