//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TORCHCONVERSION_OPS
#define TORCHCONVERSION_OPS

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "npcomp/Dialect/TorchConversion/IR/TorchConversionBase.td"
include "torch-mlir/Dialect/Torch/IR/TorchTypes.td"
include "iree-dialects/Dialect/IREE/IREEDialect.td"

class TorchConversion_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<TorchConversion_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Conversions to backend types.
//===----------------------------------------------------------------------===//

def TorchConversion_ToBuiltinTensorOp : TorchConversion_Op<"to_builtin_tensor", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Convert a `!torch.vtensor` to a `tensor`";
  let description = [{
    This op only operates on ValueTensorType, to avoid conflating conversions
    between value-semantic and non-value-semantic types.
  }];
  let arguments = (ins
    Torch_ValueTensorType:$operand
  );
  let results = (outs
    AnyTensor:$result
  );
  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];
}

def TorchConversion_FromBuiltinTensorOp : TorchConversion_Op<"from_builtin_tensor">
 {
  let summary = "Convert a `tensor` to a `!torch.vtensor`";
  let description = [{
    This op only operates on ValueTensorType, to avoid conflating conversions
    between value-semantic and non-value-semantic types.
  }];
  let arguments = (ins
    AnyTensor:$operand
  );
  let results = (outs
    Torch_ValueTensorType:$result
  );
  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];
}

def TorchConversion_ToI1Op : TorchConversion_Op<"to_i1", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Convert a `!torch.bool` to an `i1`";
  let description = [{
    This op is primarily useful as a materialization during dialect conversion.
  }];
  let arguments = (ins
    Torch_BoolType:$operand
  );
  let results = (outs
    I1:$result
  );
  let assemblyFormat = [{
    $operand attr-dict
  }];
}

def TorchConversion_FromI1Op : TorchConversion_Op<"from_i1", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Convert an `i1` to a `!torch.bool`";
  let description = [{
    This op is primarily useful as a materialization during dialect conversion.
  }];
  let arguments = (ins
    I1:$operand
  );
  let results = (outs
    Torch_BoolType:$result
  );
  let assemblyFormat = [{
    $operand attr-dict
  }];
}

def TorchConversion_ToI64Op : TorchConversion_Op<"to_i64", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Convert a `!torch.int` to an `i64`";
  let description = [{
    This op is primarily useful as a materialization during dialect conversion.
  }];
  let arguments = (ins
    Torch_IntType:$operand
  );
  let results = (outs
    I64:$result
  );
  let assemblyFormat = [{
    $operand attr-dict
  }];
}

def TorchConversion_FromI64Op : TorchConversion_Op<"from_i64", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Convert an `i64` to a `!torch.int`";
  let description = [{
    This op is primarily useful as a materialization during dialect conversion.
  }];
  let arguments = (ins
    I64:$operand
  );
  let results = (outs
    Torch_IntType:$result
  );
  let assemblyFormat = [{
    $operand attr-dict
  }];
}

def TorchConversion_ToF64Op : TorchConversion_Op<"to_f64", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Convert a `!torch.float` to an `f64`";
  let description = [{
    This op is primarily useful as a materialization during dialect conversion.
  }];
  let arguments = (ins
    Torch_FloatType:$operand
  );
  let results = (outs
    F64:$result
  );
  let assemblyFormat = [{
    $operand attr-dict
  }];
}

def TorchConversion_FromF64Op : TorchConversion_Op<"from_f64", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Convert an `f64` to a `!torch.float`";
  let description = [{
    This op is primarily useful as a materialization during dialect conversion.
  }];
  let arguments = (ins
    F64:$operand
  );
  let results = (outs
    Torch_FloatType:$result
  );
  let assemblyFormat = [{
    $operand attr-dict
  }];
}

// TODO: Verify the element types match.
def TorchConversion_ToIREEListOp : TorchConversion_Op<"to_iree_list", [
  ]> {
  let summary = "Convert a `!torch.list` to a `!iree.list`";
  let description = [{
  }];
  let arguments = (ins
    Torch_ListType:$operand
  );
  let results = (outs
    IREE_ListType:$result
  );
  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];
}

def TorchConversion_FromIREEListOp : TorchConversion_Op<"from_iree_list", [
  ]> {
  let summary = "Convert a `!iree.list` to a `!torch.list`";
  let description = [{
  }];
  let arguments = (ins
    IREE_ListType:$operand
  );
  let results = (outs
    Torch_ListType:$result
  );
  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];
}

#endif // TORCHCONVERSION_OPS
