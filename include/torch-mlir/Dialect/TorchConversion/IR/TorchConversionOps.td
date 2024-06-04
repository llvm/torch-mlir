//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHCONVERSION_OPS
#define TORCHCONVERSION_OPS

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "torch-mlir/Dialect/Torch/IR/TorchTypes.td"
include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionBase.td"

class TorchConversion_Op<string mnemonic, list<Trait> traits = []>
    : Op<TorchConversion_Dialect, mnemonic, !listconcat(traits, [Pure])> {
}

//===----------------------------------------------------------------------===//
// Conversions to backend types.
//===----------------------------------------------------------------------===//

def TorchConversion_ToBuiltinTensorOp : TorchConversion_Op<"to_builtin_tensor"> {
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
    $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
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
    $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
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
  let hasFolder = 1;
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
  let hasFolder = 1;
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
  let hasFolder = 1;
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
  let hasFolder = 1;
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
  let hasFolder = 1;
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
  let hasFolder = 1;
}

def TorchConversion_I64ToGeneratorOp : TorchConversion_Op<"i64_to_generator", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Convert an `i64` to an `Generator`";
  let description = [{
    This op is primarily useful as a materialization during dialect conversion.
  }];
  let arguments = (ins
    I64:$operand
  );
  let results = (outs
    Torch_GeneratorType:$result
  );
  let assemblyFormat = [{
    $operand attr-dict
  }];
}

def TorchConversion_GeneratorToI64Op : TorchConversion_Op<"generator_to_i64", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Convert a `Generator` to a `i64`";
  let description = [{
    This op is primarily useful as a materialization during dialect conversion.
  }];
  let arguments = (ins
    Torch_GeneratorType:$operand
  );
  let results = (outs
    I64:$result
  );
  let assemblyFormat = [{
    $operand attr-dict
  }];
}

class TorchConversionWithSideEffect_Op<string mnemonic, list<Trait> traits = []>
    : Op<TorchConversion_Dialect, mnemonic, traits> {
}

def TorchConversion_GetNextSeedOp: TorchConversionWithSideEffect_Op<"get_next_seed", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
  ]> {
  let summary = "Get the next global seed";
  let description = [{
    This op is for getting the next global seed for RNG
  }];
  let results = (outs
    I64:$result
  );
  let assemblyFormat = [{
    attr-dict `:` `(``)` `->` qualified(type($result))
  }];
}
#endif // TORCHCONVERSION_OPS
