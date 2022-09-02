//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_DIALECT_TCP_OPS
#define TORCH_MLIR_DIALECT_TCP_OPS

include "torch-mlir-dialects/Dialect/Tcp/IR/TcpBase.td"

include "mlir/IR/OpBase.td"

def Tcp_DummyOp : Tcp_Op<"dummy", []> {
  let summary = "dummy op";

  let description = [{
    Dummy identity op that returns the input as output
  }];

  let arguments = (ins
    Tcp_Tensor:$in
  );

  let results = (outs
    Tcp_Tensor:$out
  );

  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

#endif // TORCH_MLIR_DIALECT_TCP_OPS
