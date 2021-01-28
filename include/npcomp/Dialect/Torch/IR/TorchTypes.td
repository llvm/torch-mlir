//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_TYPES
#define TORCH_TYPES

include "npcomp/Dialect/Torch/IR/TorchBase.td"

//===----------------------------------------------------------------------===//
// Type defs
//===----------------------------------------------------------------------===//

class Torch_Type<string name, string typeMnemonic> : TypeDef<Torch_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Torch_NnModuleType : Torch_Type<"NnModule", "nn.Module"> {
  let summary = "torch.nn.Module";
  let description = [{
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

def AnyTorchOptionalImmutableTensor : AnyTypeOf<[
  AnyTorchImmutableTensor,
  Basicpy_NoneType,
], "allowable torch immutable tensor (or None)">;

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

def AnyTorchScalarType : AnyTypeOf<[
    AnySignedInteger,
    AnyFloat,
    Basicpy_BoolType,
    Basicpy_StrType,
    Basicpy_NoneType,
    // Allow signless integers for ease of conversions. In general, this
    // dialect uses signed integers.
    AnySignlessInteger,
  ], "Any primitive type suitable to be passed as a Torch Scalar">;

def AnyTorchBoolType : AnyTypeOf<[
    I1,
    Basicpy_BoolType,
], "Any permissible bool type">;

def AnyTorchBoolListType : AnyTypeOf<[
    Basicpy_ListType,
    // TODO: Support typed list when available.
], "Any bool list type (bool[])">;

def AnyTorchIntType : AnyTypeOf<[
    AnySignedInteger,
    AnySignlessInteger,
], "Any primitive integer type suitable to be passed as a Torch 'int'">;

def AnyTorchIntListType : AnyTypeOf<[
    Basicpy_ListType,
    // TODO: Support typed list when available.
], "Any int list type (int[])">;

def AnyTorchType : AnyTypeOf<[
    AnyTorchBoolType,
    AnyTorchScalarType,
    AnyTorchTensorType,
    Basicpy_ListType,
    Basicpy_NoneType,
    Torch_NnModuleType,
  ], "Any type that is legal to pass to a Torch kernel">;

#endif // TORCH_TYPES
