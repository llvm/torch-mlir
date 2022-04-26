//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_DIALECT_TMTENSOR_PASSES
#define TORCH_MLIR_DIALECT_TMTENSOR_PASSES

include "mlir/Pass/PassBase.td"

def TMTensorToLoops :
    Pass<"tm-tensor-to-loops", "func::FuncOp"> {
  let summary = "Convert TMTensor ops to loops and Linalg ops.";
  let constructor = "mlir::torch::TMTensor::createTMTensorToLoopsPass()";
}

def TMTensorBufferize : Pass<"tm-tensor-bufferize", "func::FuncOp"> {
  let summary = "Bufferize the TMTensor dialect";
  let constructor = "mlir::torch::TMTensor::createTMTensorBufferizePass()";
}

#endif  // TORCH_MLIR_DIALECT_TMTENSOR_PASSES
