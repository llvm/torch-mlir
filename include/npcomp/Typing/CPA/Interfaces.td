//===- CPAInterfaces.td - Interfaces to augment CPA --------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_TYPING_CPA_CPA_INTERFACES
#define NPCOMP_TYPING_CPA_CPA_INTERFACES

include "mlir/IR/OpBase.td"

def TypeMapInterface : TypeInterface<"TypeMapInterface"> {
  let methods = [
    InterfaceMethod<"Maps an MLIR Type to a CPA::TypeNode.",
      /*retTy=*/ "::mlir::NPCOMP::Typing::CPA::TypeNode *",
      /*methodName=*/ "mapToCPAType",
      /*args=*/ (ins
          "::mlir::NPCOMP::Typing::CPA::Context &":$context)>,
  ];
}

#endif // NPCOMP_TYPING_CPA_CPA_INTERFACES
