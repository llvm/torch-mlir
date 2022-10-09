//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_DIALECT_TCP_OPS
#define TORCH_MLIR_DIALECT_TCP_OPS

include "torch-mlir-dialects/Dialect/Tcp/IR/TcpBase.td"

include "mlir/IR/OpBase.td"

def Tcp_TanhOp : Tcp_UnaryElementwiseOp<"tanh"> {
  let summary = "Computes tanh of input, elementwise";

  let description = [{
    Computes elementwise hyperbolic tangent of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatOrComplexTensor:$in
  );

  let results = (outs
    Tcp_FloatOrComplexTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_AddOp : Tcp_BinaryElementwiseOp<"add", [SameOperandsAndResultElementType]> {
  let summary = "Computes elementwise addition";

  let description = [{
    Computes the elementwise addition of `in1` and `in2`.

    `in1` and `in2` must have the same shape and data type.

    A tcp.add operation will pass verification `in1`, `in2` and the result have
    compatible shapes.  Since TCP tensors have static ranks and element types,
    this means:

     * `in1`, `in2` and `out` must have the same rank.
     * `in1`, `in2` and `out` must have the same element type.
     * `in1`, `in2` and `out` must have the same size or a `?` on each dim.
  }];

  let arguments = (ins
    Tcp_Tensor:$in1,
    Tcp_Tensor:$in2
  );

  let results = (outs
    Tcp_Tensor:$out
  );

  let assemblyFormat = "$in1 `,` $in2 attr-dict `:` type($in1) `,` type($in2) `->` type($out)";
}

def Tcp_BroadcastOp : Tcp_Op<"broadcast", [
          AllRanksMatch<["in", "out"]>,
          AllElementTypesMatch<["in", "out"]>,
          PredOpTrait<
            "attribute `axes` must be in increasing order",
            CPred<"::llvm::is_sorted(getAxes(), "
                  "[](::mlir::Attribute v1, ::mlir::Attribute v2) {"
                  "return v1.cast<::mlir::IntegerAttr>().getInt() < v2.cast<::mlir::IntegerAttr>().getInt(); })">
          >,
          PredOpTrait<
            "attribute `axes` must not have any duplicates",
            CPred<"std::adjacent_find(std::begin(getAxes()), std::end(getAxes())) == std::end(getAxes())">
          >,
          PredOpTrait<
            "argument `new_dim_sizes` has the same size as the attribute `axes`",
            CPred<"getNewDimSizes().size() == getAxes().size()">
          >]> {
  let summary = "Broadcasts input to a given shape";

  let description = [{
    Broadcasts `in` to a given shape by replicating dimensions with bound `1`.

    All of the dimensions listed in `axes` must have a static size of `1` in
    `in`.  The dimesion `axis[i]` will be replicated `new_dim_sizes[i]` times in
    the output.

    This op does not change the rank of the input.

    The attribute `$axes` must be sorted and not have any duplicates. The size
    of `$axes` must be equal to the size of the variadic argument `$new_dim_sizes`.
  }];

  let arguments = (ins
    Tcp_Tensor:$in,
    Variadic<Index>:$new_dim_sizes,
    I64ArrayAttr:$axes
  );

  let results = (outs
    Tcp_Tensor:$out
  );

  let assemblyFormat = "$in `,` $new_dim_sizes attr-dict `:` type($in) `,` type($new_dim_sizes) `->` type($out)";
}

#endif // TORCH_MLIR_DIALECT_TCP_OPS
