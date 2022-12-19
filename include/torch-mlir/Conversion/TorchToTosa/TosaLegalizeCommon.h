//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_TORCHTOTOSA_TOSALEGALIZECOMMON_H
#define TORCHMLIR_CONVERSION_TORCHTOTOSA_TOSALEGALIZECOMMON_H

#include "mlir/IR/PatternMatch.h" // from @llvm-project
#include "mlir/Support/LLVM.h"    // from @llvm-project

namespace mlir {
namespace tosa {

// Lowers ReduceAll to a sequence of TOSA ops.
std::optional<Value>
convertReduceAllOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims);

// Lowers ReduceAny to a sequence of TOSA ops.
std::optional<Value>
convertReduceAnyOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims);

// Lowers ReduceMin to a sequence of TOSA ops.
std::optional<Value>
convertReduceMinOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims);

// Lowers ReduceMax to a sequence of TOSA ops.
std::optional<Value>
convertReduceMaxOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims);

// Lowers ReduceProd to a sequence of TOSA ops.
std::optional<Value>
convertReduceProdOp(PatternRewriter &rewriter, Operation *op,
                    RankedTensorType output_type, Value input_value,
                    ElementsAttr axes_elems, bool keep_dims);

// Lowers ReduceSum to a sequence of TOSA ops.
std::optional<Value>
convertReduceSumOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims);

// Lowers ReduceMean to a sequence of TOSA ops.
std::optional<Value>
convertReduceMeanOp(PatternRewriter &rewriter, Operation *op,
                    RankedTensorType output_type, Value input_value,
                    ElementsAttr axes_elems, bool keep_dims);

} // namespace tosa
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_TORCHTOTOSA_TOSALEGALIZECOMMON_H
