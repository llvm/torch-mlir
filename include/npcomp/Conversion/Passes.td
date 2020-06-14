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

//===----------------------------------------------------------------------===//
// Basicpy conversions
//===----------------------------------------------------------------------===//

def ConvertBasicpyToStd : Pass<"convert-basicpy-to-std", "FuncOp"> {
  let summary = "Convert representable Basicpy ops to std";
  let constructor = "mlir::NPCOMP::createConvertBasicpyToStdPass()";
}

//===----------------------------------------------------------------------===//
// Conditionally compiled IREE backend passes
//===----------------------------------------------------------------------===//

#ifdef NPCOMP_ENABLE_IREE
#endif // NPCOMP_ENABLE_IREE

#endif // NPCOMP_CONVERSION_PASSES
