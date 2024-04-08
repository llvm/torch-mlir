//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_TORCHONNXTOTORCH_UTILS_H
#define TORCHMLIR_CONVERSION_TORCHONNXTOTORCH_UTILS_H

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

namespace mlir::torch::onnx_c {

Value createConstantIntList(OpBinder binder,
                            ConversionPatternRewriter &rewriter,
                            SmallVector<int64_t> cstInput);

Type getQTorchTypeFromTorchIntType(Type ty);

LogicalResult OnnxLstmExpander(OpBinder binder,
                               ConversionPatternRewriter &rewriter);

bool areAllElementsDistinct(SmallVector<int64_t> array);

} // namespace mlir::torch::onnx_c

#endif // TORCHMLIR_CONVERSION_TORCHONNXTOTORCH_UTILS_H
