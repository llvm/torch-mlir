//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHCONVERSION_BASE
#define TORCHCONVERSION_BASE

include "mlir/IR/OpBase.td"

def TorchConversion_Dialect : Dialect {
  // `torch_conversion` is too verbose.
  let name = "torch_c";
  let cppNamespace = "::mlir::torch::TorchConversion";
  let description = [{
    This dialect contains ops and transforms for converting from the Torch
    backend contract to the linalg-on-tensors backend contract.

    This mainly consists of converting ops and types from `torch` dialect
    to the mix of dialects of the linalg-on-tensors backend contract, such as
    tensor ops being converted linalg-on-tensors and `!torch.vtensor` being
    converted to the builtin `tensor` type.
  }];

  let hasConstantMaterializer = 1;
}

#endif // TORCHCONVERSION_BASE
