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
      - Types specific to PyTorch.
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

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

// Torch has a fairly advanced and featureful Tensor type, and some of the
// semantics are important to preserve in a compilation context. In the future,
// a dedicated TorchTensor type may be introduced, but also, subsets of cases
// and interop are well served by existing tensor-like types, which are
// specifically permitted. Typically, on import, constraints are fairly loose
// and based on how the program is captured. Settling on and refining to
// specific types is done as part of lowering.
//
// While lowering it is useful to be able to distinguish between mutable and
// immutable tensors:
//   - Mutable tensors can alias.
//   - Mutable tensors can be a view over another mutable tensor.
//   - Mutable tensors act as if reference counted and exist for the lifetime
//     of any reference or derived view.
// Conversely, immutable tensors:
//   - Are normal SSA values representing the contents of the tensor.
//   - Cannot alias.
//   - Cannot be a view of any mutable value.
//   - Have undefined lifetimes.
//
// At the Torch dialect level, most things are modeled as an AnyTorchTensor;
// however, when lowering to specific ops, further constraints are introduced,
// necessitating copies, loads, and stores to be inserted to bridge worlds.
def AnyTorchImmutableTensor : AnyTypeOf<[
    // Normal MLIR immutable tensors.
    AnyTensor,
  ], "allowable torch immutable tensor">;

def AnyTorchMutableTensor : AnyTypeOf<[
    // "Numpy-style" mutable NDArray. While not offering the full generality
    // of a Torch tensor, it models the same access patterns and implies the
    // same aliasing as Torch tensors.
    Numpy_NdArrayType,
  ], "allowable torch mutable tensor">;

def AnyTorchTensorType : AnyTypeOf<[
    AnyTorchImmutableTensor,
    AnyTorchMutableTensor,
  ], "Any tensor type legal to pass to a Torch kernel">;

def AnyScalar : AnyTypeOf<[
    AnySignedInteger,
    Basicpy_BoolType,
    Basicpy_StrType,
    Basicpy_NoneType,
    // Allow signless integers for ease of conversions. In general, this
    // dialect uses signed integers.
    AnySignlessInteger,
  ], "Any primitive type suitable to be passed as a Torch Scalar">;

def AnyTorchType : AnyTypeOf<[
    AnyScalar,
    AnyTorchTensorType,
  ], "Any type that is legal to pass to a Torch kernel">;

#endif // TORCH_BASE
