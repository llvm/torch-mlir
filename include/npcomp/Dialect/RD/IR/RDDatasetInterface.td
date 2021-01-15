//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RD_DatasetInterface
#define RD_DatasetInterface

include "mlir/IR/OpBase.td"

def DatasetTransformOpInterface : OpInterface<"DatasetTransform"> {
  let description = [{
    Properties of dataset transformation ops; used for lowering.
  }];
  let cppNamespace = "::mlir::NPCOMP::rd";

  let methods = [
    InterfaceMethod<[{
      Returns the LLVM type used to store all state associated with the current op.
    }],
    "llvm::Optional<LLVM::LLVMType>", "buildStateLLVMType">,
    InterfaceMethod<[{
      Builds initialization code given a pointer to the type returned by buildStateLLVMType.
    }],
    "void", "buildInitState", (ins "OpBuilder &":$builder, "Value":$ptr, "const rd::InitArgMap &":$args)>,
    InterfaceMethod<[{
      Description
    }], "void", "buildNextOp",
    (ins "OpBuilder&":$builder, "BlockAndValueMapping &":$mapping, "Value":$iterator)>
  ];
}

def DatasetInitOpInterface : OpInterface<"DatasetInit"> {
  let description = [{
    Properties of dataset init ops; used for lowering.
  }];
  let cppNamespace = "::mlir::NPCOMP::rd";

  let methods = [
    InterfaceMethod<[{
      Returns the LLVM type used to store all state associated with the current op.
    }],
    "llvm::Optional<LLVM::LLVMType>", "buildStateLLVMType">
  ];
}

def DatasetNextOpInterface : OpInterface<"DatasetNext"> {
  let description = [{
    Properties of dataset next ops; used for lowering.
  }];
  let cppNamespace = "::mlir::NPCOMP::rd";

  let methods = [
    InterfaceMethod<[{
      Returns the LLVM type used to store all state associated with the current op.
    }],
    "llvm::Optional<LLVM::LLVMType>", "buildStateLLVMType">
  ];
}

#endif // #ifndef RD_DatasetInterface
