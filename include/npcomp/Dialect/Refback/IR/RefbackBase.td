//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef REFBACK_BASE
#define REFBACK_BASE

include "mlir/IR/OpBase.td"

def Refback_Dialect : Dialect {
  let name = "refback";
  let cppNamespace = "::mlir::NPCOMP::refback";
  let description = [{
    Ops used by the reference backend as part of its lowering.
  }];
}


#endif // REFBACK_BASE
