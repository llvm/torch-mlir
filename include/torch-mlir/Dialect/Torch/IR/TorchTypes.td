//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_TYPES
#define TORCH_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "torch-mlir/Dialect/Torch/IR/TorchBase.td"

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
  let hasCustomAssemblyFormat = 1;

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
  let hasCustomAssemblyFormat = 1;
}

// For standard ArrayRefs, which require allocation.
class OptionalArrayRefTorchParameter<string arrayOf, string desc = ""> :
    AttrOrTypeParameter<
      "::std::optional<::llvm::ArrayRef<" # arrayOf # ">>", desc> {
  let allocator = [{
    if ($_self.has_value()) {
      $_dst.value() = $_allocator.copyInto($_self.value());
    }
  }];
}

class AnyTorchTensorType<string name, string typeMnemonic>
    : Torch_Type<name, typeMnemonic, "::mlir::torch::Torch::BaseTensorType"> {
  let summary = "Multi-dimensional array modeling Torch's Tensor type";
  let description = [{
    Syntax:

    ```
    tensor-type ::= (`!torch.tensor` | `!torch.vtensor`) tensor-modifiers?
    tensor-modifiers ::= `<` sizes-spec `,` dtype-spec (',' sparsity)? `>`
    sizes-spec ::= `*` | `[` size-list `]`
    size-list ::= /*empty*/ | size-list-nonempty
    size-list-nonempty = size (`,` size)*
    size ::= `?` | decimal-literal
    dtype-spec ::= `unk` | type
    sparsity ::= attribute-value
    ```

    Represents a multi-dimensional array to model Torch's `torch.Tensor` type.

    If the type is `!torch.tensor`, it represents a general unrestricted
    `torch.Tensor`, including potential mutability, aliasing, etc.
    If the type is `!torch.vtensor` then the tensor is restricted to operations
    that have value semantics ("v" = "value semantics"). This helps to maintain
    a strict separation between the value-semantic and potentially-mutating
    worlds, as one of our main jobs in Torch-MLIR is to normalize the program
    into a form with value semantics.
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
    | torch.quint8      | !torch.quint8      |
    | torch.complex64   | complex<f32>       |
    | torch.complex128  | complex<f64>       |
    |-------------------|--------------------|
    ```

    The `sparsity` attribute directly mirrors the additional tensor `encoding`
    defined by upstream MLIR on the RankedTensorType. Unlike the upstream
    attribute, however, this attribute is exclusively used to denote a
    straightforward tensor (with an empty attribute) or a sparse tensor
    (with a SparseTensorEncodingAttr).

    TODO: Support the full set of Torch dtypes.
    TODO: Use si1?

    Note: We avoid the C++ identifier `TensorType` to avoid C++ name ambiguities
    with `mlir::TensorType`, since most code is transitively nested in
    both `::mlir` and `::mlir::torch::Torch` namespaces.

    Note: We use the Torch-aligned terminology "sizes" and "dtype" instead of
    the MLIR-aligned terminology "rank/shape" and "element type". The cheat
    sheet is:
    - `hasRank()` -> `hasSizes()`
    - `getShape()` -> `getSizes()`
    - `getElementType()` -> `getDtype()` (but be sure that `hasDtype()` though).
  }];
  let parameters = (ins
    OptionalArrayRefTorchParameter<"int64_t", "sizes of dimensions">:$optionalSizes,
    "::mlir::Type":$optionalDtype,
    "Attribute":$optionalSparsity
  );
  let builders = [
    // Provide builder where optionalSparsity is empty by default.
    TypeBuilder<(ins
      "::std::optional<ArrayRef<int64_t>>":$optionalSizes,
      "::mlir::Type":$optionalDtype,
      CArg<"Attribute", "{}">:$optionalSparsity
    ), [{
      return $_get(context, optionalSizes, optionalDtype, optionalSparsity);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
  string extraBaseClassDeclaration = [{
  }];
}

def Torch_NonValueTensorType : AnyTorchTensorType<"NonValueTensor", "tensor"> {
  let extraClassDeclaration = extraBaseClassDeclaration # [{
    // Get this type, with value semantics added.
    ValueTensorType getWithValueSemantics() const;
    // Get the !torch.tensor type with the least static information.
    static NonValueTensorType getWithLeastStaticInformation(MLIRContext *context);
  }];
}

def Torch_ValueTensorType : AnyTorchTensorType<"ValueTensor", "vtensor"> {
  let extraClassDeclaration = extraBaseClassDeclaration # [{
    // Get this type, with value semantics removed.
    NonValueTensorType getWithoutValueSemantics() const;
    // Get the !torch.tensor type with the least static information.
    static ValueTensorType getWithLeastStaticInformation(MLIRContext *context);
    // Get the builtin tensor type with the same static information as this one,
    // or nullptr if that is not possible (i.e. when the dtype is unknown).
    TensorType toBuiltinTensor() const;
  }];
}

def AnyTorchTensorType : Type<
    CPred<"$_self.isa<::mlir::torch::Torch::BaseTensorType>()">,
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

def Torch_TupleType : Torch_Type<"Tuple", "tuple"> {
  let summary = "!torch.tuple<T1, T2, T3>";
  let description = [{
    Tuple type with 0-N ordered contained types.
  }];
  let parameters = (ins
    ArrayRefParameter<"::mlir::Type", "contained types">:$containedTypes
  );
  let hasCustomAssemblyFormat = 1;
}

def Torch_UnionType : Torch_Type<"Union", "union"> {
  let summary = "!torch.union<T1, T2, T3>";
  let description = [{
    Union type with 0-N alternative types.

    NOTE: We use the terminology "contained types" for consistency with
    PyTorch. Strictly speaking, the types aren't "contained" though.

    TODO: Canonicalize unions based on subtype relations, to allow
    using pointer equality to compare two unions for being the same.
    For now, `!torch.union<T1, T2>` is different from `!torch.union<T2, T1>`,
    and same for `!torch.union<T1, SubtypeOfT1>` vs `!torch.union<T1>`.
  }];
  let parameters = (ins
    ArrayRefParameter<"::mlir::Type", "contained types">:$containedTypes
  );
  let hasCustomAssemblyFormat = 1;
}

def Torch_DeviceType : Torch_Type<"Device", "Device"> {
  let summary = "Torch device";
}

def Torch_GeneratorType : Torch_Type<"Generator", "Generator"> {
  let summary = "Torch Generator for producing valsem-random numbers";
}

def Torch_BoolType : Torch_Type<"Bool", "bool"> {
  let summary = "Torch BoolType";
  let description = [{
    An immutable boolean taking values 0 or 1.
  }];
}

def Torch_IntType : Torch_Type<"Int", "int"> {
  let summary = "Torch IntType";
  let description = [{
    The integer type used to model the Python `int` type in TorchScript.
    TorchScript itself models this type as a 64-bit signed integer.

    Note: This type is not used for modeling tensor dtypes.
  }];
}

def Torch_FloatType : Torch_Type<"Float", "float"> {
  let summary = "Torch FloatType";
  let description = [{
    The float type is used to model the Python `float` type in TorchScript.
    Python and TorchScript use 64-bit floating point for this type at runtime.

    Note: This type is not used for modeling tensor dtypes.
  }];
}

def Torch_NoneType : Torch_Type<"None", "none"> {
  let summary = "Torch NoneType";
  let description = [{
    The singleton "None" type.
  }];
}

def Torch_StringType : Torch_Type<"String", "str"> {
  let summary = "Torch StringType";
  let description = [{
    An immutable string representing a sequence of characters.

    TODO: Figure out the exact TorchScript/PyTorch string semantics.
    E.g. is it always unicode-encoded, etc.
  }];
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

def Torch_QUInt8Type : Torch_Type<"QUInt8", "quint8"> {
  let summary = "Type modeling `ScalarType::QUInt8`";
  let description = [{
    This is intended to be a 1:1 match for the Torch `ScalarType` types.

    Looking at the variety / ad-hocness (e.g. `QUInt4x2`) of that set of
    types, it is deemed preferable to import them as one-off ad-hoc types
    instead of a single parameterized type.
  }];
}

def Torch_QInt32Type : Torch_Type<"QInt32", "qint32"> {
  let summary = "Type modeling `ScalarType::QInt32`";
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
    for backends.
  }];
}

def Torch_AnyType : Torch_Type<"Any", "any"> {
  let summary = "Torch any type";
  let description = [{
    Represent any torch type. All the other types are sub types of Any type.
  }];
}

def Torch_NumberType : Torch_Type<"Number", "number"> {
  let summary = "Torch number type";
  let description = [{
    The Int, Float and Complex type are sub types of Number type.
  }];
}

def Torch_DictType : Torch_Type<"Dict", "dict"> {

  let summary = "!torch.dict[KT, VT] ";
  let parameters = (ins "::mlir::Type":$keyType, "::mlir::Type":$valueType);
  let description = [{
    Torch Dict type with key and value type.
  }];
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$keyType,
                                        "::mlir::Type":$valueType), [{
      return Base::get(keyType.getContext(), keyType, valueType);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

class OptionalOf<Type type, string descr> :
    AnyTypeOf<[type, Torch_OptionalType, Torch_NoneType], descr> ;

def AnyTorchOptionalTensorType :
      OptionalOf<AnyTorchTensorType, "Optional torch tensor type">;
def AnyTorchOptionalNonValueTensorType :
      OptionalOf<Torch_NonValueTensorType, "Optional torch non value tensor type">;
def AnyTorchOptionalIntType: OptionalOf<Torch_IntType, "Optional torch int type">;
def AnyTorchOptionalFloatType: OptionalOf<Torch_FloatType, "Optional torch float type">;
def AnyTorchOptionalBoolType:
      OptionalOf<Torch_BoolType, "Optional torch bool type">;
def AnyTorchOptionalStringType:
      OptionalOf<Torch_StringType, "Optional torch Str type">;
def AnyTorchOptionalDeviceType:
      OptionalOf<Torch_DeviceType, "Optional torch device type">;
def AnyTorchOptionalGeneratorType:
      OptionalOf<Torch_GeneratorType, "Optional torch Generator type">;

def IsListTypePred : CPred<"$_self.isa<::mlir::torch::Torch::ListType>()">;
class ListOf<list<Type> allowedTypes, string descr> :
    ContainerType<AnyTypeOf<allowedTypes>,
            IsListTypePred,
            "$_self.cast<::mlir::torch::Torch::ListType>().getContainedType()",
            descr, "::mlir::torch::Torch::ListType">;

def AnyTorchListOfTorchBoolType : ListOf<[Torch_BoolType], "Bool list type (bool[])">;
def AnyTorchListOfTorchIntType : ListOf<[Torch_IntType], "Int list type (int[])">;
def AnyTorchListOfTorchFloatType : ListOf<[Torch_FloatType], "Float list type (float[])">;
def AnyTorchListOfTorchStringType : ListOf<[Torch_StringType], "Str list type (str[])">;
def AnyTorchListOfTensorType:
      ListOf<[AnyTorchTensorType], "Any int list type (Tensor[])">;
def AnyTorchListOfNonValueTensorType:
      ListOf<[Torch_NonValueTensorType], "Any int list type (Tensor[])">;
def AnyTorchListOfOptionalTensorType :
      ListOf<[AnyTorchOptionalTensorType],
             "Any optional tensor list type (Tensor?[])">;
def AnyTorchListOfOptionalNonValueTensorType :
      ListOf<[AnyTorchOptionalNonValueTensorType],
             "Any optional tensor list type (Tensor?[])">;
def AnyTorchListOfOptionalIntType :
      ListOf<[AnyTorchOptionalIntType],
             "List of optional ints type (int?[])">;
def AnyTorchOptionalListOfTorchIntType : OptionalOf<AnyTorchListOfTorchIntType, "Optional torch int list type (int[]?)">;
def AnyTorchOptionalListOfTorchFloatType : OptionalOf<AnyTorchListOfTorchFloatType, "Optional torch float list type (float[]?)">;
// Note: TorchScript does not consider !torch.bool to be a Scalar.
def AnyTorchScalarType :
    Type<CPred<"isValidSubtype($_self, ::mlir::torch::Torch::NumberType::get($_self.getContext()))">,
         "Any Python numeric type compatible with being the scalar type of a tensor">;
def AnyTorchOptionalScalarType:
      OptionalOf<AnyTorchScalarType, "Optional torch scalar type">;

// See function `DictTypePtr create(TypePtr key, TypePtr value)`
// in aten/src/ATen/core/jit_type.h.
def AnyTorchDictKeyType : AnyTypeOf<[
  Torch_AnyType,
  Torch_IntType,
  Torch_BoolType,
  Torch_FloatType,
  Torch_StringType,
  AnyTorchTensorType,
], "Allowed dict key types">;

// In alphabetic order.
def AnyTorchType : AnyTypeOf<[
  AnyTorchScalarType,
  AnyTorchTensorType,
  Torch_AnyType,
  Torch_BoolType,
  Torch_DictType,
  Torch_DeviceType,
  Torch_GeneratorType,
  Torch_ListType,
  Torch_LinearParamsType,
  Torch_NumberType,
  Torch_NnModuleType,
  Torch_NoneType,
  Torch_OptionalType,
  Torch_StringType,
  Torch_TupleType,
  Torch_UnionType,
], "Any type that is legal to pass to a Torch kernel">;

def AnyTorchListType : ListOf<[AnyType], "Any Torch list Type">;

#endif // TORCH_TYPES
