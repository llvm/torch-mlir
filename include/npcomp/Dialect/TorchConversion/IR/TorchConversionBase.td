//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TORCHCONVERSION_BASE
#define TORCHCONVERSION_BASE

include "mlir/IR/OpBase.td"

def TorchConversion_Dialect : Dialect {
  // `torch_conversion` is too verbose.
  let name = "torch_c";
  let cppNamespace = "::mlir::NPCOMP::TorchConversion";
  let description = [{
    This dialect contains ops and transforms for converting from the Torch
    backend contract to the npcomp backend contract.

    This mainly consists of converting ops and types from `torch` dialect
    to the mix of dialects of the npcomp backend contract, such as tensor
    ops being converted linalg-on-tensors, lists being converted to IREE lists,
    and !torch.float being converted to `f64`.
  }];
}

#endif // TORCHCONVERSION_BASE
