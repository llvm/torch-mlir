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

def Tcp_ClampOp : Tcp_UnaryElementwiseOp<"clamp"> {
  let summary = "Clamps input tensor to the given min and/or max";

  let description = [{
    Clamps the input tensor, elementwise, to the given min and/or max values.

    The min and max values specified as attributes should correspond to the
    element type of the input tensor.
     * If the input is a float tensor, only `min_float` and/or `max_float` must
       be specified.
     * If the input is an int tensor, only `min_int` and/or `max_int` must be
       specified.
  }];

  // TODO: Does clamp need to support complex tensors?
  let arguments = (ins
    Tcp_FloatOrIntTensor:$in,
    OptionalAttr<F32Attr>:$min_float,
    OptionalAttr<F32Attr>:$max_float,
    OptionalAttr<I64Attr>:$min_int,
    OptionalAttr<I64Attr>:$max_int
  );

  let results = (outs
    Tcp_FloatOrIntTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";

  let hasVerifier = 1;
}

def Tcp_AddOp : Tcp_BinaryElementwiseOp<"add", [SameOperandsAndResultElementType]> {
  let summary = "Computes elementwise addition";

  let description = [{
    Computes the elementwise addition of `in1` and `in2`.
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

def Tcp_SubOp : Tcp_BinaryElementwiseOp<"sub", [SameOperandsAndResultElementType]> {
  let summary = "Computes elementwise addition";

  let description = [{
    Computes the elementwise subtraction of `in2` from `in1`.
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

def Tcp_MulOp : Tcp_BinaryElementwiseOp<"mul", [SameOperandsAndResultElementType]> {
  let summary = "Computes elementwise multiplication";

  let description = [{
    Computes the elementwise multiplication of `in1` and `in2`.
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
          AllElementTypesMatch<["in", "out"]>]> {
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

  let hasVerifier = 1;
}

#endif // TORCH_MLIR_DIALECT_TCP_OPS
