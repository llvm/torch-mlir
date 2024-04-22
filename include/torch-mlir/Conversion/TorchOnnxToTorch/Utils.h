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

#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

class Endian {
private:
  static constexpr uint32_t uint32_ = 0x01020304;
  static constexpr uint8_t magic_ = (const uint8_t &)uint32_;

public:
  static constexpr bool little = magic_ == 0x04;
  static constexpr bool big = magic_ == 0x01;
  static_assert(little || big, "Cannot determine endianness!");

private:
  Endian() = delete;
};

namespace mlir::torch::onnx_c {

Value createConstantIntList(OpBinder binder,
                            ConversionPatternRewriter &rewriter,
                            SmallVector<int64_t> cstInput);

Type getQTorchTypeFromTorchIntType(Type ty);

LogicalResult OnnxLstmExpander(OpBinder binder,
                               ConversionPatternRewriter &rewriter);

bool areAllElementsDistinct(SmallVector<int64_t> array);

namespace detail {
/// Matches the constant integers stored in a `onnx.Constant`.
struct onnx_list_of_constant_ints_op_binder {
  SmallVectorImpl<int64_t> &bind_values;

  /// Creates a matcher instance that binds the value to bvs if match succeeds.
  onnx_list_of_constant_ints_op_binder(SmallVectorImpl<int64_t> &bvs)
      : bind_values(bvs) {}

  bool match(Operation *op) {
    auto constOp = dyn_cast<Torch::OperatorOp>(op);
    if (!constOp || !constOp.getName().equals("onnx.Constant"))
      return false;

    if (DenseResourceElementsAttr attr =
            constOp->getAttr("torch.onnx.value")
                .dyn_cast_or_null<DenseResourceElementsAttr>()) {
      // Bytes are stored in little endian order. Big endian support will
      // require swizzling.
      if (!Endian::little) {
        op->emitError("unimplemented: importing on big endian systems");
        return false;
      }

      auto ty = cast<ShapedType>(attr.getType());
      ElementsAttr denseAttr;
      auto ptr = attr.getRawHandle().getBlob()->getData();
      denseAttr = DenseElementsAttr::getFromRawBuffer(ty, ptr);
      for (auto axis : denseAttr.getValues<llvm::APInt>()) {
        bind_values.push_back(axis.getSExtValue());
      }
      return true;
    }
    return false;
  }
};
} // namespace detail

/// Matches the constant integers stored in a `onnx.Constant`.
inline detail::onnx_list_of_constant_ints_op_binder
m_OnnxListOfConstantInts(SmallVectorImpl<int64_t> &bind_values) {
  return detail::onnx_list_of_constant_ints_op_binder(bind_values);
}

std::optional<int64_t> onnxDtypeIntToTorchDtypeInt(int64_t dtypeIntOnnx);

} // namespace mlir::torch::onnx_c

#endif // TORCHMLIR_CONVERSION_TORCHONNXTOTORCH_UTILS_H
