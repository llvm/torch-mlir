//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_DIALECTS_CONVERSION_PASSES
#define TORCH_MLIR_DIALECTS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

#ifdef TORCH_MLIR_DIALECTS_ENABLE_TCP
//===----------------------------------------------------------------------===//
// TcpToLinalg
//===----------------------------------------------------------------------===//

def ConvertTcpToLinalg
      : Pass<"convert-tcp-to-linalg", "func::FuncOp"> {
  let summary = "Lower TCP to Linalg";
  let description = [{
    Pass that converts TCP operations to equivalent operations in Linalg.
  }];

  let constructor = "mlir::tcp::createConvertTcpToLinalgPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
}
#endif // TORCH_MLIR_DIALECTS_ENABLE_TCP

#endif // TORCH_MLIR_DIALECTS_CONVERSION_PASSES