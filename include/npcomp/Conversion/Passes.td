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
// ATen conversions
//===----------------------------------------------------------------------===//

def ConvertATenToTCF : Pass<"convert-aten-to-tcf", "FuncOp"> {
  let summary = "Convert recognized ATen to TCF ops";
  let constructor = "mlir::NPCOMP::createConvertATenToTCFPass()";
}

//===----------------------------------------------------------------------===//
// Basicpy conversions
//===----------------------------------------------------------------------===//

def ConvertBasicpyToStd : Pass<"convert-basicpy-to-std", "FuncOp"> {
  let summary = "Convert representable Basicpy ops to std";
  let constructor = "mlir::NPCOMP::createConvertBasicpyToStdPass()";
}

//===----------------------------------------------------------------------===//
// Numpy conversions
//===----------------------------------------------------------------------===//

def ConvertNumpyToTCF : Pass<"convert-numpy-to-tcf", "FuncOp"> {
  let summary = "Convert the numpy dialect to supported TCF ops";
  let constructor = "mlir::NPCOMP::createConvertNumpyToTCFPass()";
}

//===----------------------------------------------------------------------===//
// TCFToTCP
//===----------------------------------------------------------------------===//

def ConvertTCFToLinalg : Pass<"convert-tcf-to-linalg", "ModuleOp"> {
  let summary = "Convert TCF to Linalg";
  let description = [{
    The intention is for this pass to convert mainly to linalg named ops.

    Because linalg is at the "TCP" layer of abstraction, this pass has to
    concern itself with generating guards for error cases.
  }];
  let constructor = "mlir::NPCOMP::createConvertTCFToLinalgPass()";
}

//===----------------------------------------------------------------------===//
// TCFToStd
//===----------------------------------------------------------------------===//

def ConvertTCFToStd : Pass<"convert-tcf-to-std", "ModuleOp"> {
  let summary = "Convert TCF to Std";
  let constructor = "mlir::NPCOMP::createConvertTCFToStdPass()";
}

//===----------------------------------------------------------------------===//
// TCFToTCP
//===----------------------------------------------------------------------===//

def ConvertTCFToTCP : Pass<"convert-tcf-to-tcp", "ModuleOp"> {
  let summary = "Convert TCF to TCP";
  let constructor = "mlir::NPCOMP::createConvertTCFToTCPPass()";
}

//===----------------------------------------------------------------------===//
// Conditionally compiled IREE backend passes
//===----------------------------------------------------------------------===//

#ifdef NPCOMP_ENABLE_IREE
#endif // NPCOMP_ENABLE_IREE

#endif // NPCOMP_CONVERSION_PASSES
