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
include "torch-mlir-dialects/Dialect/Tcp/IR/TcpEnums.td"

include "mlir/IR/OpBase.td"

def Tcp_TanhOp : Tcp_UnaryElementwiseOp<"tanh", [SameOperandsAndResultElementType]> {
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

def Tcp_ClampOp : Tcp_UnaryElementwiseOp<"clamp", [SameOperandsAndResultElementType]> {
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

def Tcp_SigmoidOp : Tcp_UnaryElementwiseOp<"sigmoid", [SameOperandsAndResultElementType]> {
  let summary = "Computes sigmoid of input, elementwise";

  let description = [{
    Computes elementwise sigmoid of the input tensor.
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
  let summary = "Computes elementwise subtraction";

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
    Tcp_FloatOrIntTensor:$in1,
    Tcp_FloatOrIntTensor:$in2
  );

  let results = (outs
    Tcp_FloatOrIntTensor:$out
  );

  let assemblyFormat = "$in1 `,` $in2 attr-dict `:` type($in1) `,` type($in2) `->` type($out)";
}

def Tcp_DivFOp : Tcp_BinaryElementwiseOp<"divf", [SameOperandsAndResultElementType]> {
  let summary = "Computes elementwise floating point division";

  let description = [{
    Computes the elementwise floating point division of `in1` and `in2`.
  }];

  let arguments = (ins
    Tcp_FloatTensor:$in1,
    Tcp_FloatTensor:$in2
  );

  let results = (outs
    Tcp_FloatTensor:$out
  );

  let assemblyFormat = "$in1 `,` $in2 attr-dict `:` type($in1) `,` type($in2) `->` type($out)";
}

def Tcp_ConstOp : Tcp_Op<"const", [ConstantLike, Pure]> {
  let summary = "Constant op";

  let description = [{
    An op that produces constant tensor (of floating point or signless integer type).
  }];

  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    Tcp_Tensor:$out
  );

  let assemblyFormat = "attr-dict `:` type($out)";

  let hasFolder = 1;
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

def Tcp_YieldOp : Tcp_Op<"yield", [Terminator, Pure]> {
  let summary = "Yields values from regions in Tcp";

  let description = [{
    Terminates and yields values from regions in Tcp.

    The values that are yielded should correspond to the parent operation's
    results.

    The results can be `AnyType` since Tcp grouping ops can be used with ops
    from any dialect (not just Tcp).
  }];

  let arguments = (ins
    Variadic<AnyType>:$ins
  );

  let builders = [
    OpBuilder<(ins), [{ }]>,
  ];

  let assemblyFormat = "$ins attr-dict `:` type($ins)";
}

def Tcp_GroupOp : Tcp_Op<"group", [
          AffineScope,
          ParentOneOf<["func::FuncOp", "tcp::GroupOp", "tcp::IsolatedGroupOp"]>,
          SingleBlockImplicitTerminator<"tcp::YieldOp">]> {
  let summary = "Groups ops into a region that is not isolated from above";

  let description = [{
    Provides a way to group ops into a region that is not isolated from above.

    Since this is not isolated from above, ops inside this region can refer to
    values in the outer scopes. So, this op does not have any input parameters.

    This op must not include control flow ops in its region. However, it can
    be present within regions that are inside control flow ops.

    The enclosed region will terminate with a `tcp.yield` op. This ops results
    should correspond to the values that are yielded by the terminating op.

    The results can be `AnyType` since Tcp grouping ops can be used with ops
    from any dialect (not just Tcp).
  }];

  let results = (outs
    Variadic<AnyType>:$outs
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let assemblyFormat = "attr-dict-with-keyword $body `:` type($outs)";

  let hasVerifier = 1;
}

def Tcp_IsolatedGroupOp : Tcp_Op<"isolated_group", [
          AffineScope,
          IsolatedFromAbove,
          ParentOneOf<["func::FuncOp", "tcp::GroupOp", "tcp::IsolatedGroupOp"]>,
          SingleBlockImplicitTerminator<"tcp::YieldOp">]> {
  let summary = "Groups ops into a region that is isolated from above";

  let description = [{
    Provides a way to group ops into a region that is isolated from above.

    Since the enclosed region is isolated from above, any use of values from
    outer scopes requires passing that value as an input parameter to this op
    and using that parameter instead.

    This op must not include control flow ops in its region. However, it can
    be present within regions that are inside control flow ops.

    The enclosed region will terminate with a `tcp.yield` op. This ops results
    should correspond to the values that are yielded by the terminating op.

    The inputs and results can be `AnyType` since Tcp grouping ops can be used
    with ops from any dialect (not just Tcp).
  }];

  let arguments = (ins
    Variadic<AnyType>:$ins
  );

  let results = (outs
    Variadic<AnyType>:$outs
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let assemblyFormat = "$ins attr-dict-with-keyword $body `:` type($ins) `->` type($outs)";

  let hasVerifier = 1;
}

def Tcp_SqrtOp : Tcp_UnaryElementwiseOp<"sqrt"> {
  let summary = "Computes square root of input, elementwise";

  let description = [{
    Computes elementwise square root of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatOrIntTensor:$in
  );

  let results = (outs
    Tcp_FloatTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_ConcatOp : Tcp_Op<"concat", [SameOperandsAndResultElementType]> {
  let summary = "Concatenates input tensors";

  let description = [{
    Concatenates tensors of same datatype and rank into a single tensor on specific single axis.
  }];

  let arguments = (ins
    Variadic<Tcp_Tensor>:$inputs,
    I64Attr:$axis
  );

  let results = (outs
    Tcp_Tensor:$result
  );

  let assemblyFormat = "$inputs attr-dict-with-keyword `:` type($inputs) `->` type($result)";

  let hasVerifier = 1;
}

def Tcp_CeilOp : Tcp_UnaryElementwiseOp<"ceil", [SameOperandsAndResultElementType]> {
  let summary = "Computes ceil of input, elementwise";

  let description = [{
    Computes elementwise ceil of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatOrComplexTensor:$in
  );

  let results = (outs
    Tcp_FloatOrComplexTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_FloorOp : Tcp_UnaryElementwiseOp<"floor", [SameOperandsAndResultElementType]> {
  let summary = "Computes floor of input, elementwise";

  let description = [{
    Computes elementwise floor of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatOrComplexTensor:$in
  );

  let results = (outs
    Tcp_FloatOrComplexTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_CosOp : Tcp_UnaryElementwiseOp<"cos", [SameOperandsAndResultElementType]> {
  let summary = "Computes cosine of input, elementwise";

  let description = [{
    Computes elementwise cosine of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatOrComplexTensor:$in
  );

  let results = (outs
    Tcp_FloatOrComplexTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_SinOp : Tcp_UnaryElementwiseOp<"sin", [SameOperandsAndResultElementType]> {
  let summary = "Computes sine of input, elementwise";

  let description = [{
    Computes elementwise sine of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatOrComplexTensor:$in
  );

  let results = (outs
    Tcp_FloatOrComplexTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_AbsOp : Tcp_UnaryElementwiseOp<"abs", [SameOperandsAndResultElementType]> {
  let summary = "Computes absolute of input, elementwise";

  let description = [{
    Computes elementwise absolute value of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatOrIntTensor:$in
  );

  let results = (outs
    Tcp_FloatOrIntTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_LogOp : Tcp_UnaryElementwiseOp<"log", [SameOperandsAndResultElementType]> {
  let summary = "Computes natural logarithm of input, elementwise";

  let description = [{
    Computes elementwise natural logarithm value of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatOrComplexTensor:$in
  );

  let results = (outs
    Tcp_FloatOrComplexTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_NegOp : Tcp_UnaryElementwiseOp<"neg", [SameOperandsAndResultElementType]> {
  let summary = "Computes the negation of input, elementwise";

  let description = [{
    Computes elementwise negation value of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatOrComplexTensor:$in
  );

  let results = (outs
    Tcp_FloatOrComplexTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_AtanOp : Tcp_UnaryElementwiseOp<"atan", [SameOperandsAndResultElementType]> {
  let summary = "Computes the arcus tangent value of input, elementwise";

  let description = [{
    Computes elementwise arcus tangent value of the input tensor.
  }];

  let arguments = (ins
    Tcp_FloatOrComplexTensor:$in
  );

  let results = (outs
    Tcp_FloatOrComplexTensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tcp_Atan2Op : Tcp_BinaryElementwiseOp<"atan2", [SameOperandsAndResultElementType]> {
  let summary = "2-argument arcus tangent of the given values";

  let description = [{
    It is a generalization of the 1-argument arcus tangent which returns the angle 
    on the basis of the ratio in1 / in2, elementwise.
  }];

  let arguments = (ins
    Tcp_FloatOrComplexTensor:$in1,
    Tcp_FloatOrComplexTensor:$in2
  );

  let results = (outs
    Tcp_FloatOrComplexTensor:$out
  );

  let assemblyFormat = "$in1 `,` $in2 attr-dict `:` type($in1) `,` type($in2) `->` type($out)";
}

def Tcp_CastOp:  Tcp_Op<"cast", [Pure, Elementwise, SameOperandsAndResultShape]> {

  let summary = "TCP Cast operation";

  // TODO: Make the cast op handle more corner cases
  // e.g. onversion overflows the destination type
  // e.g. convert negative number to unsigned destination type
  let description = [{
    Performs elementwise type conversion.

    Supported MLIR core types:
      INT: ui8,  si8,  i8, 
           ui16, si16, i16,
           ui32, si32, i32,
           ui64, si64, i64, 
           i1 (Boolean)
      FP: f16, bf16, f32, f64

    Supported Conversion Mode:
      FP -> FP
      INT -> FP
      FP -> INT
      INT -> INT
  }];

  let arguments = (ins
    Tcp_Tensor:$in,
    OptionalAttr<Tcp_SignednessAttr>:$in_int_signedness,
    OptionalAttr<Tcp_SignednessAttr>:$out_int_signedness
  );

  let results = (outs
    Tcp_Tensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";

  let hasVerifier = 1;
}

#endif // TORCH_MLIR_DIALECT_TCP_OPS
