//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_TCP_PASSES
#define NPCOMP_TCP_PASSES

include "mlir/Pass/PassBase.td"

def TCPBufferize : Pass<"tcp-bufferize", "FuncOp"> {
  let summary = "Bufferizes the tcp dialect";
  let constructor = "mlir::NPCOMP::createTCPBufferizePass()";
}

#endif // NPCOMP_TCP_PASSES
