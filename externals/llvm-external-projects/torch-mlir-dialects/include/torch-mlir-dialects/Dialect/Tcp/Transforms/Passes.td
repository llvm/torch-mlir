//===-- Passes.td - Pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_TCP_PASSES
#define TORCHMLIR_TCP_PASSES

include "mlir/Pass/PassBase.td"

// \brief This is a pass that performs fusion of TCP ops.
def TcpFuseElementwiseOps : Pass<"tcp-fuse-elementwise-ops", "ModuleOp"> {
  let summary = "Performs fusion of tcp elementwise ops";
  let constructor = "mlir::tcp::createTcpFuseElementwiseOpsPass()";
}

// \brief This pass makes all TCP group ops isolated from above.
def TcpIsolateGroupOps : Pass<"tcp-isolate-group-ops", "ModuleOp"> {
  let summary = "Converts all tcp.group ops to tcp.isolated_group ops";
  let constructor = "mlir::tcp::createTcpIsolateGroupOpsPass()";
}

#endif // TORCHMLIR_TCP_PASSES
