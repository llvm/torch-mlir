//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_CONVERSION_PASSES
#define NPCOMP_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TCFToTCP
//===----------------------------------------------------------------------===//

def ConvertTCFToTCP : Pass<"convert-tcf-to-tcp", "ModuleOp"> {
  let summary = "Convert TCF to TCP";
  let constructor = "mlir::NPCOMP::createConvertTCFToTCPPass()";
}

//===----------------------------------------------------------------------===//
// TCPToLinalg
//===----------------------------------------------------------------------===//

def ConvertTCPToLinalg : Pass<"convert-tcp-to-linalg", "ModuleOp"> {
  let summary = "Convert TCP to Linalg";
  let constructor = "mlir::NPCOMP::createConvertTCPToLinalgPass()";
}

#endif // NPCOMP_CONVERSION_PASSES
