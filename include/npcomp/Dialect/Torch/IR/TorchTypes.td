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
    Represents an instance of a `torch.nn.Module` with the given `className`.
  }];
  let parameters = (ins StringRefParameter<"class name">:$className);

  let printer = [{
    $_printer << "nn.Module<\"";
    llvm::printEscapedString(getImpl()->className, $_printer.getStream());
    $_printer << "\">";
  }];

  let parser = [{
    if (parser.parseLess())
      return Type();
    StringRef className;
    if ($_parser.parseOptionalString(&className))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, className);
  }];
}

// TODO: It feels like this should be something more general.
// However, to do that, we need to agree on construction operations
// and the valid MLIR representations of the "None" state.
//
// For now, we only need it as a stand-in type to allow importing
// the `_is_full_backward_hook` optional bool type that Torch puts on
// all classes.
def Torch_OptionalType : Torch_Type<"Optional", "optional"> {
  let summary = "!torch.optional<T>";
  let description = [{
  }];
  let parameters = (ins "::mlir::Type":$containedType);

  let printer = [{
    $_printer << "optional<" << getImpl()->containedType << ">";
  }];

  let parser = [{
    if (parser.parseLess())
      return Type();
    Type containedType;
    if ($_parser.parseType(containedType))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, containedType);
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$containedType), [{
      return Base::get(containedType.getContext(), containedType);
    }]>
  ];
}

def Torch_DeviceType : Torch_Type<"Device", "Device"> {
  let summary = "Torch device";
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

def AnyTorchOptionalTensor : AnyTypeOf<[
  AnyTorchTensorType,
  Torch_OptionalType,
  Basicpy_NoneType,
], "optional torch tensor">;

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

def AnyTorchNumberType : AnyTypeOf<[
    AnySignedInteger,
    AnyFloat,
    Basicpy_BoolType,
    // Allow signless integers for ease of conversions. In general, this
    // dialect uses signed integers.
    AnySignlessInteger,
  ], "Any primitive numeric type">;

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
    Basicpy_TupleType,
    Basicpy_NoneType,
    Basicpy_BytesType,
    Torch_NnModuleType,
    Torch_OptionalType,
    Torch_DeviceType,
  ], "Any type that is legal to pass to a Torch kernel">;

#endif // TORCH_TYPES
