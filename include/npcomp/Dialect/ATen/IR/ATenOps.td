//===- ATen.td ---------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_ATEN_IR_ATEN_OPS
#define NPCOMP_DIALECT_ATEN_IR_ATEN_OPS

include "npcomp/Dialect/ATen/IR/ATenDialect.td"
include "npcomp/Dialect/ATen/IR/ATenOpInterface.td"
include "npcomp/Dialect/Torch/IR/OpInterfaces.td"
include "npcomp/Dialect/Torch/IR/TorchTypes.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

// TODO: convert to "let results =" style
// TODO: Rename prefix from "aten" to "ATen" for consistency.

class aten_Op<string mnemonic, list<OpTrait> traits = [StatisticsOpInterface]> :
    Op<ATen_Dialect, mnemonic, traits>;


// Most ops are automatically generated from pytorch specs.
include "npcomp/Dialect/ATen/IR/GeneratedATenOps.td"
include "npcomp/Dialect/ATen/IR/LegacyGeneratedATenOps.td"

def aten_BatchNormOp: aten_Op<"batch_norm", [NoSideEffect, StatisticsOpInterface]>,
                      Results<(outs AnyTensor:$output, AnyTensor:$save_mean, AnyTensor:$save_invstd)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2,
        AnyType:$arg3,
        AnyType:$arg4,
        AnyType:$arg5,
        AnyType:$arg6,
        AnyType:$arg7,
        AnyType:$arg8
  );

  let summary = "BatchNorm operator";
  let description = [{
    BatchNorm operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

// We have list constants, which come out of pytorch.  Represent them using
// our own constant-like type, which gets lowered to std_ConstantOp later.
def aten_ConstantOp: aten_Op<"constant", [NoSideEffect]>,
                     Results<(outs AnyType)> {
  let summary = "Constant operator";
  let description = [{
    Constant operator
  }];

}

def aten_FlattenOp: aten_Op<"flatten", [NoSideEffect, StatisticsOpInterface]>,
                    Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2
  );

  let summary = "Flatten operator";
  let description = [{
    Flatten operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_MaxPool2dOp: aten_Op<"max_pool2d", [NoSideEffect, StatisticsOpInterface]>,
                      Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyType:$arg0,
        AnyType:$arg1,
        AnyType:$arg2,
        AnyType:$arg3,
        AnyType:$arg4,
        AnyType:$arg5
  );

  let summary = "MaxPool2d operator";
  let description = [{
    MaxPool2d operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
  }];
}

def aten_TypeCastOp : aten_Op<"type_cast", [NoSideEffect]>,
                      Results<(outs AnyType)> {
  let summary = "TypeCast operator";
  let arguments = (
    ins AnyType:$x
  );
}

#endif // NPCOMP_DIALECT_ATEN_IR_ATEN_OPS
