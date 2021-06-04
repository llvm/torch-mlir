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

class Torch_Type<string name, string typeMnemonic,
                 string baseCppClass = "::mlir::Type">
    : TypeDef<Torch_Dialect, name, [], baseCppClass> {
  let mnemonic = typeMnemonic;
}

class Torch_TypeWithContainedType<string name, string typeMnemonic> : Torch_Type<name, typeMnemonic> {
  let parameters = (ins "::mlir::Type":$containedType);

  let printer = [{
    $_printer << getMnemonic() << "<" << getImpl()->containedType << ">";
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

// For standard ArrayRefs, which require allocation.
class OptionalArrayRefParameter<string arrayOf, string desc = ""> :
    AttrOrTypeParameter<
      "::llvm::Optional<::llvm::ArrayRef<" # arrayOf # ">>", desc> {
  let allocator = [{
    if ($_self.hasValue()) {
      $_dst.getValue() = $_allocator.copyInto($_self.getValue());
    }
  }];
}

class AnyTorchTensorType<string name, string typeMnemonic>
    : Torch_Type<name, typeMnemonic, "::mlir::NPCOMP::Torch::BaseTensorType"> {
  let summary = "Multi-dimensional array modeling Torch's Tensor type";
  let description = [{
    Syntax:

    ```
    tensor-type ::= (`!torch.tensor` | `!torch.vtensor`) tensor-modifiers?
    tensor-modifiers ::= `<` sizes-spec `,` dtype-spec `>`
    sizes-spec ::= `*` | `[` size-list `]`
    size-list ::= /*empty*/ | size-list-nonempty
    size-list-nonempty = size (`,` size)*
    size ::= `?` | decimal-literal
    dtype-spec ::= `unk` | type
    ```

    Represents a multi-dimensional array to model Torch's `torch.Tensor` type.

    If the type is `!torch.tensor`, it represents a general unrestricted
    `torch.Tensor`, including potential mutability, aliasing, etc.
    If the type is `!torch.vtensor` then the tensor is restricted to operations
    that have value semantics ("v" = "value semantics"). This helps to maintain
    a strict separation between the value-semantic and potentially-mutating
    worlds, as one of our main jobs in the compiler is to isolate the mutating
    parts as much as possible because most lower levels of the compiler stack
    are expected to require value semantics. E.g. npcomp's backend contract
    is mostly in terms of linalg-on-tensor for compute-heavy ops, which require
    a conversion to the builtin `tensor` type which has value semantics.
    Some notes about value semantics:
      - Using the type system described in PEP 483 (which TorchScript and other
        Python systems follow), `!torch.tensor` is a subtype of
        `!torch.vtensor`. Specifically, both types have the same set of values,
        but `!torch.tensor` additionally allows aliasing or mutating
        operations.
      - Despite being a subtype, a `!torch.tensor` carries *less* static
        information than a corresponding `!torch.vtensor`. In particular,
        `!torch.vtensor` carries the static information "not used in aliasing
        or mutating operations".
      - `!torch.vtensor` can be trivially converted to the builtin `tensor`
        type when the dtype is known (the builtin `tensor` type does not allow
        an unknown dtype).

    In the absence of the `tensor-modifiers`, the type contains the minimal
    amount of static information. That is, `!torch.tensor` is equivalent to
    `!torch.tensor<*,unk>` (and similarly for `!torch.vtensor`).

    If `sizes-spec` is not `*`, it indicates additional static information
    about the sizes of the tensor. It will consist of a list of elements,
    with length equal to the "rank" (in MLIR parlance) or "ndim"
    (in Torch parlance). Each element represents a size, with the typical
    MLIR representation of a number for a statically known size and `?` for a
    size that is unknown. Thus, the lattice consists of `*` as the least static
    information, followed by lists containing unknown sizes such as `[?,?,?]`
    which contribute rank information, followed by statically specified sizes
    for some dimensions such as `[?,3,?]`, followed by fully statically
    specified sizes such as `[2,3,4]`.

    If `dtype-spec` is not `unk` ("unknown"), it contains an MLIR type
    which contributes static information about the dtype of the tensor.
    Only types allowed by Torch are permitted.
    ```
    |-------------------|--------------------|
    | Torch Type        | MLIR Type          |
    |-------------------|--------------------|
    | torch.float16     | f16                |
    | torch.bfloat16    | bf16               |
    | torch.float32     | f32                |
    | torch.float64     | f64                |
    | torch.uint8       | ui8                |
    | torch.int8        | si8                |
    | torch.int16       | si16               |
    | torch.int32       | si32               |
    | torch.int64       | si64               |
    | torch.bool        | i1                 |
    | torch.qint8       | !torch.qint8       |
    |-------------------|--------------------|
    ```

    TODO: Support the full set of Torch dtypes.
    TODO: Use si1?

    Note: We avoid the C++ identifier `TensorType` to avoid C++ name ambiguities
    with `mlir::TensorType`, since most code is transitively nested in
    both `::mlir` and `::mlir::NPCOMP::Torch` namespaces.

    Note: We use the Torch-aligned terminology "sizes" and "dtype" instead of
    the MLIR-aligned terminology "rank/shape" and "element type". The cheat
    sheet is:
    - `hasRank()` -> `hasSizes()`
    - `getShape()` -> `getSizes()`
    - `getElementType()` -> `getDtype()` (but be sure that `hasDtype()` though).
  }];
  let parameters = (ins
    OptionalArrayRefParameter<"int64_t", "sizes of dimensions">:$optionalSizes,
    "::mlir::Type":$optionalDtype
  );
  let genVerifyDecl = 1;
  string extraBaseClassDeclaration = [{
  }];
}

def Torch_NonValueTensorType : AnyTorchTensorType<"NonValueTensor", "tensor"> {
  let extraClassDeclaration = extraBaseClassDeclaration # [{
    // Get this type, with value semantics added.
    ValueTensorType getWithValueSemantics() const;
    // Get the !torch.tensor type with the least static information.
    static NonValueTensorType getWithLeastStaticInformation(MLIRContext *context);
    // Get a NonValueTensorType with shape/dtype matching `type`.
    static NonValueTensorType getFromShaped(ShapedType type);
  }];
}

def Torch_ValueTensorType : AnyTorchTensorType<"ValueTensor", "vtensor"> {
  let extraClassDeclaration = extraBaseClassDeclaration # [{
    // Get this type, with value semantics removed.
    NonValueTensorType getWithoutValueSemantics() const;
    // Get the !torch.tensor type with the least static information.
    static ValueTensorType getWithLeastStaticInformation(MLIRContext *context);
    // Get a NonValueTensorType with shape/dtype matching `type`.
    static ValueTensorType getFromShaped(ShapedType type);
    // Get the builtin tensor type with the same static information as this one,
    // or nullptr if that is not possible (i.e. when the dtype is unknown).
    TensorType toBuiltinTensor() const;
  }];
}

def AnyTorchTensorType : Type<
    CPred<"$_self.isa<::mlir::NPCOMP::Torch::BaseTensorType>()">,
    "Any Torch tensor type"
>;

// TODO: It feels like this should be something more general.
// However, to do that, we need to agree on construction operations
// and the valid MLIR representations of the "None" state.
//
// For now, we only need it as a stand-in type to allow importing
// the `_is_full_backward_hook` optional bool type that Torch puts on
// all classes.
def Torch_OptionalType : Torch_TypeWithContainedType<"Optional", "optional"> {
  let summary = "!torch.optional<T>";
  let description = [{
  }];
}

def Torch_ListType : Torch_TypeWithContainedType<"List", "list"> {
  let summary = "!torch.list<T>";
  let description = [{
  }];
}

def Torch_DeviceType : Torch_Type<"Device", "Device"> {
  let summary = "Torch device";
}

def Torch_QInt8Type : Torch_Type<"QInt8", "qint8"> {
  let summary = "Type modeling `ScalarType::QInt8`";
  let description = [{
    This is intended to be a 1:1 match for the Torch `ScalarType` types.

    Looking at the variety / ad-hocness (e.g. `QUInt4x2`) of that set of
    types, it is deemed preferable to import them as one-off ad-hoc types
    instead of a single parameterized type.
  }];
}

def Torch_LinearParamsType : Torch_Type<"LinearParams", "LinearParams"> {
  let summary = "Torch packed linear params type";
  let description = [{
    A weight and optional bias, packed into one value.

    This is used to model the
    `__torch__.torch.classes.quantized.LinearPackedParamsBase` custom C++ class
    type which is the input to some Torch `quantized::` ops.

    We may want to eventually have a full set of ops that model the
    LinearPackedParamsBase interface, such as `apply`, `apply_relu`, etc.
    But we instead are likely to just expand the `quantized::` ops directly
    and fold away the instances of this type.
    The whole LinearPackedParamsBase abstraction as it stands in PyTorch is a
    very library-call-y, runtime-y thing that embodies a number of assumptions
    about the structure of how the program will be executed, which need not hold
    for npcomp backends.
  }];
}

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

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

def IsListTypePred : CPred<"$_self.isa<::mlir::NPCOMP::Torch::ListType>()">;

class ListOf<list<Type> allowedTypes, string descr > :
          ContainerType<AnyTypeOf<allowedTypes>, IsListTypePred,
            "$_self.cast<::mlir::NPCOMP::Torch::ListType>().getContainedType()",
              descr, "::mlir::NPCOMP::Torch::ListType">;

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

def AnyTorchBoolListType : ListOf<[AnyTorchBoolType], "Any bool list type (bool[])">;

def AnyTorchIntType : AnyTypeOf<[
    AnySignedInteger,
    AnySignlessInteger,
], "Any primitive integer type suitable to be passed as a Torch 'int'">;

def AnyTorchIntListType : ListOf<[AnyTorchIntType], "Any int list type (int[])">;

def AnyTorchType : AnyTypeOf<[
    AnyTorchBoolType,
    AnyTorchScalarType,
    AnyTorchTensorType,
    Basicpy_TupleType,
    Basicpy_NoneType,
    Basicpy_BytesType,
    Torch_NnModuleType,
    Torch_OptionalType,
    Torch_ListType,
    Torch_DeviceType,
    Torch_LinearParamsType,
  ], "Any type that is legal to pass to a Torch kernel">;

def AnyTorchListType : ListOf<[AnyType], "Any Torch list Type">;

#endif // TORCH_TYPES
