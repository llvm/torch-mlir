//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_BASE
#define TORCH_BASE

include "mlir/IR/OpBase.td"
include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.td"
include "npcomp/Dialect/Numpy/IR/NumpyDialect.td"

def Torch_Dialect : Dialect {
  let name = "torch";
  let cppNamespace = "::mlir::NPCOMP::Torch";
  let description = [{
    Top-level dialect for interfacing PyTorch and MLIR.

    This dialect contains types and structural ops that model enough of
    PyTorch's behavior to allow for easy import/call-out. While not aiming to
    be completely isomorphic, it is laid out to make conversion in/out
    systematic for the supported features (some of which are aspirational):
      - Transitions between mutable and immutable tensors.
      - Gradient associations and management.
      - Custom ops.
      - Types specific to PyTorch such as torch.nn.Module structures
      - Module level constructs like quantization parameters, globals, etc.

    Where possible, this dialect composes with types and ops from the `Numpy`
    and `Basicpy` dialects, and those dialects should be considered "upstream"
    for basic Python and ND-Array based programming constructs.

    As a key point, this dialect does not contain any custom operations,
    including those that people would typically associate as core (see
    the `ATen` dialect for mathematical ops like add, conv, etc), instead
    modeling the open op-system that PyTorch reasons about natively.
  }];
}

#endif // TORCH_BASE
