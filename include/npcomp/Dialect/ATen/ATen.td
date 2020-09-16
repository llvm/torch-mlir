//===- ATen.td ---------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

#ifndef ATEN_OPS
#define ATEN_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "npcomp/Dialect/ATen/ATenOpInterface.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

/// The ATenDialect models 'A Tensor library' from Pytorch.  The intention
/// is to provide an abstraction which is isomorphic with datastructures
/// returned from the pytorch jit, enabling integration with Pytorch models.
/// Most of the actual operation definitions in tablegen are themselves
/// generated from C APIs exported by Pytorch.
def ATen_Dialect : Dialect {
  let name = "aten";
  let cppNamespace = "::mlir::NPCOMP::aten";
}

//===----------------------------------------------------------------------===//
// Dialect types
//===----------------------------------------------------------------------===//

def ATen_ListType : DialectType<ATen_Dialect,
    CPred<"$_self.isa<::mlir::NPCOMP::aten::ATenListType>()">, "ATen List">,
    BuildableType<"$_builder.getType<::mlir::NPCOMP::aten::ATenListType()"> {
  let typeDescription = [{
    A variadic list of arguments in ATen.
  }];
}

// TODO: convert to "let results =" style
// TODO: Rename prefix from "aten" to "ATen" for consistency.

class aten_Op<string mnemonic, list<OpTrait> traits = [StatisticsOpInterface]> :
    Op<ATen_Dialect, mnemonic, traits>;


// Most ops are automatically generated from pytorch specs.
include "npcomp/Dialect/ATen/ATenOps.td"


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

// Our jit library only supports 6 argument convolutions, rather than 9
// arguments supported by pytorch.  This operation allows us to represent this
// limitation temporarily.
def aten_ConvolutionOp: aten_Op<"_convolution", [NoSideEffect, StatisticsOpInterface]>,
                        Results<(outs AnyTensor)> {
  let arguments = (
    ins AnyTensor:$input,
        AnyTensor:$weight,
        AnyTensor:$bias,
        AnyType:$stride,
        AnyType:$padding,
        AnyType:$dilation
  );

  let summary = "Convolution operator";
  let description = [{
    Convolution operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
    uint64_t getOperandTransferVolume(unsigned int idx, bool read);
    uint64_t getResultTransferVolume(unsigned int idx, bool read);
	}];
}

// Our jit library only supports 6 argument convolutions, rather than 9
// arguments supported by pytorch.  This operation allows us to represent this
// limitation temporarily.
def aten_ConvolutionBackwardOp: aten_Op<"_convolution_backward", [NoSideEffect, StatisticsOpInterface]>,
                                Results<(outs AnyTensor:$dx, AnyTensor:$dw, AnyTensor:$db)> {
  let arguments = (
  ins AnyTensor:$grad_output,
      AnyTensor:$input,
      AnyTensor:$weight,
      AnyType:$stride,
      AnyType:$padding,
      AnyType:$dilation
  );

  let summary = "ConvolutionBackward operator";
  let description = [{
    ConvolutionBackward operator
  }];
  let extraClassDeclaration = [{
    std::map<std::string, uint64_t> getStatistics();
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

#endif
