//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_OPS
#define TORCH_OPS

include "npcomp/Dialect/Torch/IR/TorchTypes.td"
include "npcomp/Interfaces/Traits.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Torch_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Torch_Dialect, mnemonic, traits> {
}

include "npcomp/Dialect/Torch/IR/GeneratedAtenOps.td"
include "npcomp/Dialect/Torch/IR/GeneratedPrimOps.td"
include "npcomp/Dialect/Torch/IR/GeneratedQuantizedOps.td"

//===----------------------------------------------------------------------===//
// TorchScript `torch.nn.Module` object instantiation ops.
//===----------------------------------------------------------------------===//

def Torch_NnModuleOp : Torch_Op<"nn_module", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    SingleBlockImplicitTerminator<"::mlir::NPCOMP::Torch::NnModuleTerminatorOp">]> {
  let summary = "Constructs a torch.nn.Module";
  let description = [{
    This op is used to represent a torch.nn.Module when importing a
    graph of Python objects.

    This op returns a new torch.nn.Module as an SSA value, with a set of
    declaratively specified properties.

    Example:

    ```mlir
    %2 = torch.nn_module {
      torch.slot "b", %bool_true : !basicpy.BoolType
      torch.slot "i", %num3_i64 : i64
      torch.slot "f", %num : f64
      torch.slot "t", %t : !torch.tensor
      torch.slot "submodule", %1 : !torch.nn.Module
    } : !torch.nn.Module<"my_class_name">
    ```

    This op is tightly coupled to the `torch.class_type` op named in the
    `!torch.nn.Module<"my_class_name">` type. Each slot must match precisely
    with the corresponding `torch.attr` in the `torch.class_type`.
    See the documentation for `torch.class_type` for information.
  }];

  let arguments = (ins);
  let results = (outs Torch_NnModuleType:$result);
  let regions = (region SizedRegion<1>:$region);
  let verifier = "return ::verify(*this);";

  let assemblyFormat = "$region attr-dict `:` type($result)";

  let extraClassDeclaration = [{
    StringRef getClassName() { return getType().getClassName(); }
    ClassTypeOp getClassType(::mlir::SymbolTable &symbolTable) {
      return symbolTable.lookup<ClassTypeOp>(getClassName());
    }
  }];
}

def Torch_NnModuleTerminatorOp : Torch_Op<"nn_module_terminator", [Terminator,
    HasParent<"::mlir::NPCOMP::Torch::NnModuleOp">]> {
  let summary = "Implicit terminator for torch.nn_module";

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "attr-dict";
}

def Torch_SlotOp : Torch_Op<"slot", [
    HasParent<"::mlir::NPCOMP::Torch::NnModuleOp">]> {
  let summary = "Define the value of a slot of a torch.nn.Module";
  let description = [{
    This op specifies that the initial value of the slot `name` of the
    parent torch.nn_module should be `value`, which is allowed to be an
    arbitrary Torch-compatible SSA value, including other !torch.nn.Module's.
  }];

  let arguments = (ins StrAttr:$name, AnyTorchType:$value);
  let results = (outs);

  let assemblyFormat = [{
    $name `,` $value attr-dict `:` type($value)
  }];
}

//===----------------------------------------------------------------------===//
// Modeling of TorchScript class types
//===----------------------------------------------------------------------===//

def Torch_ClassTypeOp : Torch_Op<"class_type", [
    Symbol,
    SingleBlockImplicitTerminator<"::mlir::NPCOMP::Torch::ClassTypeTerminatorOp">]> {
  let summary = "Constructs a torch.ClassType";
  let description = [{
    Declares a class type. Class types are the types used to describe
    TorchScript `torch.nn.Module`'s. The terminology "class type" is for
    consistency with TorchScript (a better name in our context might be
    "nn module subtype"). The `syn_name` of this op is the same string
    as in the `!torch.nn.Module<"...">` type.

    Example:

    ```mlir
    // A simple empty torch.class_type, with corresponding torch.nn_module.
    torch.class_type @empty {}
    %submodule = torch.nn_module {} : !torch.nn.Module<"empty">

    // A class type with many members.
    torch.class_type @test {
      torch.attr "b" : !basicpy.BoolType
      torch.attr "i" : i64
      torch.attr "f" : f64
      torch.attr "t" : !torch.tensor
      torch.attr "submodule" : !torch.nn.Module<"empty">
      torch.method "method", @f
    }
    torch.nn_module {
      // These must match the order and names in the `torch.class_type`.
      torch.slot "b", %bool_true : !basicpy.BoolType
      torch.slot "i", %num3_i64 : i64
      torch.slot "f", %num : f64
      torch.slot "t", %t : !torch.tensor
      torch.slot "submodule", %submodule : !torch.nn.Module<"empty">
    } : !torch.nn.Module<"test">
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let results = (outs);
  let regions = (region SizedRegion<1>:$region);
  let verifier = "return ::verify(*this);";

  let assemblyFormat = "$sym_name $region attr-dict";
}

def Torch_ClassTypeTerminatorOp : Torch_Op<"class_type_terminator", [Terminator,
    HasParent<"::mlir::NPCOMP::Torch::ClassTypeOp">]> {
  let summary = "Implicit terminator for torch.class_type";

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "attr-dict";
}

def Torch_MethodOp : Torch_Op<"method", [
    HasParent<"::mlir::NPCOMP::Torch::ClassTypeOp">,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Declare a method of a torch.class_type";
  let description = [{
    This op declaratively specifies that the parent torch.class_type has a
    method `name` which calls `function`. `function` is an unbound function.
    That is, it explicitly takes the torch.nn.Module as a parameter (no implicit
    "self" object).

    If `private` is present, it indicates that external calls cannot be made
    to this method.
  }];

  // We don't use sym_visibility because that only applies to Symbol's, and
  // some of the related concepts like "nested" visibility are specific to
  // symbols.
  let arguments = (ins
    StrAttr:$name,
    FlatSymbolRefAttr:$function,
    // `private` is a C++ keyword, so use `isPrivate`.
    UnitAttr:$isPrivate
  );
  let results = (outs);

  let assemblyFormat = [{
    (`private` $isPrivate^)? $name `,` $function attr-dict
  }];
}

def Torch_AttrOp : Torch_Op<"attr", [
    HasParent<"::mlir::NPCOMP::Torch::ClassTypeOp">
  ]> {
  let summary = "Declare an attribute of a torch.class_type";
  let description = [{
    This op declaratively specifies that torch.nn.Module's of the parent
    torch.class_type must have an attribute `name` of type `type`.

    If `private` is present, it indicates that the value of this attribute
    cannot be accessed externally.
  }];

  // We don't use sym_visibility because that only applies to Symbol's, and
  // some of the related concepts like "nested" visibility are specific to
  // symbols.
  let arguments = (ins
    StrAttr:$name,
    TypeAttr:$type,
    // `private` is a C++ keyword, so use `isPrivate`
    UnitAttr:$isPrivate
  );
  let results = (outs);

  let assemblyFormat = [{
    (`private` $isPrivate^)? $name `:` $type attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Global slot ops
//===----------------------------------------------------------------------===//
// TODO: Should these be in a separate dialect?
// At this point, they are fairly specific to torch types, but their get/set
// semantics follow Python.
//===----------------------------------------------------------------------===//

def Torch_GlobalSlotOp : Torch_Op<"global_slot", [
    Symbol,
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"::mlir::NPCOMP::Torch::GlobalSlotInitOp">
  ]> {
  let summary = "A slot with global storage";
  let description = [{
    Represents a slot with global storage. The slot semantics are the same
    as Python's: getting or setting a slot is done by object identity.

    The `typeBound` is a type that the contained type is a subtype of.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    TypeAttr:$typeBound
  );
  let results = (outs);
  let regions = (region SizedRegion<1>:$initializer);

  let assemblyFormat = [{
    ($sym_visibility^)? $sym_name attr-dict `:` $typeBound ($initializer^)?
  }];
}

def Torch_GlobalSlotInitOp : Torch_Op<"global_slot.init", [
    Terminator,
    HasParent<"::mlir::NPCOMP::Torch::GlobalSlotOp">]> {
  let summary = "yield-like terminator for torch.global_slot initializer region";
  let description = [{
    The operand to this op becomes the initial value of the parent
    torch.global_slot.
  }];

  let arguments = (ins AnyTorchType:$initialValue);
  let results = (outs);

  // This bulider creates an illegal op, but is needed to appease
  // ensureTerminator in the default builders for SingleBlockImplicitTerminator
  // on the parent torch.global_slot op.
  // TODO: Have a SingleBlockExplicitTerminator trait.
  let builders = [OpBuilder<(ins), [{ /*nothing to do */ }]>];

  let assemblyFormat = "$initialValue attr-dict `:` type($initialValue)";
}

def Torch_GlobalSlotGetOp : Torch_Op<"global_slot.get", []> {
  let summary = "Get the value stored in a torch.global_slot";

  let arguments = (ins
    FlatSymbolRefAttr:$slot
  );
  let results = (outs AnyTorchType:$result);

  let assemblyFormat = [{
    $slot attr-dict `:` type($result)
  }];
}

def Torch_GlobalSlotSetOp : Torch_Op<"global_slot.set", []> {
  let summary = "Set the value stored in a torch.global_slot";

  let arguments = (ins
    FlatSymbolRefAttr:$slot,
    AnyTorchType:$value
  );
  let results = (outs);

  let assemblyFormat = [{
    $slot `=` $value attr-dict `:` type($value)
  }];
}

//===----------------------------------------------------------------------===//
// TorchScript interpreter builtin ops.
//===----------------------------------------------------------------------===//
// These don't correspond to a `torch::jit::Operator`, so they don't appear
// in the registry and cannot be autogenerated.
// Most of these correspond 1:1 to interpreter opcodes, though some
// (like control flow being lowered to raw branches) are not directly mapped.
// See `torch/csrc/jit/runtime/instruction.h`.


def Torch_PrimListUnpackOp: Torch_Op<"prim.ListUnpack",
    [AllowsTypeRefinement]> {
  let summary = "TorchScript prim::ListUnpack op";
  let arguments = (ins AnyTorchType:$operand);
  let results = (outs Variadic<AnyTorchType>:$results);

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($results)
  }];
}

def Torch_PrimTupleConstructOp: Torch_Op<"prim.TupleConstruct", [
    NoSideEffect,
    TypesMatchWith<"contained types correspond to operand types",
    "elements", "result", "Torch::TupleType::get($_ctxt, llvm::to_vector<6>($_self))">
    ]> {
  let summary = "TorchScript prim::TupleConstruct op";
  let description = [{
    Note: This op does not allow trivial type refinement, because the
    operand types and the result types must be in correspondence.
  }];

  let arguments = (ins
    Variadic<AnyTorchType>:$elements
  );
  let results = (outs
    Torch_TupleType:$result
  );

  let assemblyFormat = [{
    $elements attr-dict `:` type($elements)
  }];
}

def Torch_PrimListConstructOp: Torch_Op<"prim.ListConstruct", [
    NoSideEffect,
    AllowsTypeRefinement,
    ]> {
  let summary = "TorchScript prim::ListConstruct op";

  let arguments = (ins
    Variadic<AnyTorchType>:$elements
  );
  let results = (outs
    AnyTorchListType:$result
  );

  let verifier = "return ::verify(*this);";

  let assemblyFormat = [{
    $elements attr-dict `:` functional-type(operands, results)
  }];
}

def Torch_PrimGetAttrOp : Torch_Op<"prim.GetAttr", []> {
  let summary = "TorchScript prim::GetAttr op";

  let arguments = (ins StrAttr:$name, Torch_NnModuleType:$receiver);
  let results = (outs AnyTorchType:$result);

  let assemblyFormat = [{
    $receiver `[` $name `]` attr-dict `:` type($receiver) `->` type($result)
  }];
}

def Torch_PrimSetAttrOp : Torch_Op<"prim.SetAttr", []> {
  let summary = "TorchScript prim::SetAttr op";

  let arguments = (ins
    StrAttr:$name,
    Torch_NnModuleType:$receiver,
    AnyTorchType:$value
  );
  let results = (outs);

  let assemblyFormat = [{
    $receiver `[` $name `]` `=` $value attr-dict `:` type($receiver) `,` type($value)
  }];
}

def Torch_PrimCallMethodOp : Torch_Op<"prim.CallMethod", []> {
  let summary = "TorchScript prim::CallMethod op";

  let arguments = (ins
    StrAttr:$name,
    Torch_NnModuleType:$receiver,
    Variadic<AnyTorchType>:$operands
  );
  let results = (outs AnyTorchType:$result);

  let assemblyFormat = [{
    $receiver `[` $name `]` `(` $operands `)` attr-dict `:` type($receiver) `,` functional-type($operands, $result)
  }];
}

def Torch_PrimLoopOp : Torch_Op<"prim.Loop", [
  DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorEntryOperands"]>]> {
  let summary = "TorchScript prim::Loop op";
  let description = [{
    This op (together with prim.Loop.condition) define a looping construct
    that combines `for` and `while` behavior.

    See: https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md#loops
  }];

  let arguments = (ins
    I64:$maxTripCount,
    Basicpy_BoolType:$initialCondition,
    Variadic<AnyTorchType>:$iterArgsInit
  );
  let results = (outs Variadic<AnyTorchType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = [{
    $maxTripCount `,` $initialCondition `,` `init` `(` $iterArgsInit `)` $region
    attr-dict `:` functional-type(operands, results)
  }];
  let verifier = [{ return RegionBranchOpInterface::verifyTypes(*this); }];
}

def Torch_PrimLoopConditionOp : Torch_Op<"prim.Loop.condition", [
    Terminator,
    HasParent<"::mlir::NPCOMP::Torch::PrimLoopOp">]> {
  let summary = "yield-like terminator for torch.prim.Loop";
  let description = [{
    Does not correspond to any torch prim op directly (the way that they model
    blocks has a built-in notion of yield-like terminator).
  }];

  let arguments = (ins
    Basicpy_BoolType:$shouldContinue,
    Variadic<AnyTorchType>:$iterArgs
  );
  let results = (outs);

  let assemblyFormat = [{
    $shouldContinue `,`
    `iter` `(` ($iterArgs^ `:` type($iterArgs))? `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Additional ops used to model TorchScript's Graph's / Node's.
//===----------------------------------------------------------------------===//

def Torch_DerefineOp : Torch_Op<"derefine", [
    NoSideEffect,
    DeclareOpInterfaceMethods<CastOpInterface>,
  ]> {
  let summary = "De-refine a type";
  let description = [{
    In terms of IR structure, TorchScript allows types to vary in many
    circumstances where MLIR requires pointer-identical types. In particular,
    it is valid to pass any subtype in place of a type. For example, if an
    `Optional[int]` is required somewhere in the IR, it is legal to pass a
    value of just `int` (but not the other way around; see
    `torch.prim.unchecked_cast`). In effect, every *use* can have a different
    type.

    This op bridges that impedance mismatch. This op allows casting a value
    from one type to a type that it is a subtype of to model this behavior.
    This op uses the TorchScript notion of subtype, which matches the
    Python notion of subtype presented in PEP 483.
  }];

  let arguments = (ins AnyTorchType:$operand);
  let results = (outs AnyTorchType:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `to` type($result)
  }];

  let hasCanonicalizer = 1;
}

def Torch_OperatorOp : Torch_Op<"operator", [
    AllowsTypeRefinement
  ]> {
  let summary = "Opaque torch operator";
  let description = [{
    Represents an invocation of a `torch::jit::Operator` for which we don't
    have a registered MLIR operation.

    The `name` attribute contains the name that the MLIR op would have
    (excluding `torch.`) if we did have it registered, which allows easy
    cross referencing with `JITOperatorRegistryDump.txt`.
  }];

  let arguments = (ins StrAttr:$name, Variadic<AnyTorchType>:$operands);
  let results = (outs Variadic<AnyTorchType>:$results);

  let assemblyFormat = [{
    $name `(` $operands `)` attr-dict `:` functional-type($operands, $results)
  }];
}

def Torch_LinearParamsCreateOp : Torch_Op<"linear_params.create", [
    AllowsTypeRefinement
  ]> {
  let summary = "Create a `!torch.LinearParams`";
  let arguments = (ins
    AnyTorchTensorType:$weight,
    Optional<AnyTorchTensorType>:$bias
  );
  let results = (outs Torch_LinearParamsType:$result);

  let assemblyFormat = [{
    $weight (`,` $bias^)? attr-dict `:` type($weight) (`,` type($bias)^)?
  }];
}

def Torch_PerTensorAffineCreateOp : Torch_Op<"per_tensor_affine.create", [
    AllowsTypeRefinement
  ]> {
  let summary = "Create a per-tensor-affine quantized tensor";
  let description = [{
    Create a quantized tensor.

    Quantization formula is:
    ```
    Q(x, scale, zero_point) = round(x/scale + zero_point)
    ```

    See:
    https://pytorch.org/docs/stable/quantization.html#quantized-tensors
  }];
  let arguments = (ins
    AnyTorchTensorType:$int_repr,
    AnyFloat:$scale,
    AnyTorchIntType:$offset
  );
  // TODO: Limit to quantized dtypes (e.g. !torch.qint8).
  let results = (outs AnyTorchTensorType:$result);

  let assemblyFormat = [{
    $int_repr `,` $scale `,` $offset attr-dict
    `:` type($int_repr) `,` type($scale) `,` type($offset) `->` type($result)
  }];
}

// TODO: Disaggregate this op into a value-semantic constant + val->nonval
// conversion if needed.
// Currently, this op can effectively hide val->nonval conversion, which makes
// it an edge case for passes that care about that such as
// torch-maximize-value-semantics.
// So the suggestion would be to lower this to a `torch.vtensor` op
// (+`torch.copy.tensor` if needed).
// In particular, currently we end up relying on convert-torch-to-std
// to effectively expose this (as part of lowering to `std.constant`) +
// hoping that some canonicalization cleans it up.
// The `torch-maximize-value-semantics` pass should be doing this
// before we convert to std at all.
def Torch_TensorOp : Torch_Op<"tensor", [
    DeclareOpInterfaceMethods<InferTypeOpInterface, ["isCompatibleReturnTypes"]>,
    AllowsTypeRefinement
  ]> {
  let summary = "Create a value of !torch.tensor type from a literal";
  let description = [{
    Example:
    ```
    %0 = torch.tensor(dense<0.0> : tensor<3x5xf32>) : !torch.tensor
    %1 = torch.tensor(dense<0.0> : tensor<3xf32>) : !torch.vtensor<[3],f32>
    ```
  }];
  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyTorchTensorType:$result);

  let assemblyFormat = [{
    `(` $value `)` attr-dict `:` type($result)
  }];

  let extraClassDeclaration = [{
    // InferTypeOpInterface:
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual);
  }];
}

def Torch_TensorStaticInfoCastOp : Torch_Op<"tensor_static_info_cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    AllowsTypeRefinement,
    NoSideEffect]> {
  let summary = "Adds/removes static information from a tensor type.";
  let description = [{
    This op does not imply any runtime code. Semantically it is an identity
    function. However, it statically annotates (or erases) shape and dtype
    information from a tensor type.

    This op *cannot* be used to add/remove value semantics from a tensor.
    For converting between the value-semantic and non-value-semantic domains,
    use `torch.copy.tensor`. The two ops are kept separate to prevent
    canonicalizations from accidentally dropping static information. In
    most cases, after running the `torch-refine-types` pass, this op becomes
    a no-op (the pass will incorporate the static information into other ops
    that allow type refinement).
  }];
  let arguments = (ins
    AnyTorchTensorType:$operand
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `to` type($result)
  }];
}

def Torch_CopyTensorOp : Torch_Op<"copy.tensor", []> {
  let summary = "Makes a copy of a tensor.";
  let description = [{
    Changes to the original tensor will not be reflected in the copy.

    This op can be used to interconvert between value-semantic and
    non-value-semantic tensors. However, this op *does not* allow
    adding/removing static information about sizes/dtype. For that, use
    `torch.tensor_static_info_cast`.

    This op does not have the AllowsTypeRefinement trait because the operand
    and result types are coupled. Only places that know how to simultaneously
    update both types should be changing the type of this op.
  }];
  let arguments = (ins
    AnyTorchTensorType:$operand
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];
  let verifier = "return ::verify(*this);";
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Torch_OverwriteTensorOp : Torch_Op<"overwrite.tensor", [
    AllowsTypeRefinement
  ]> {
  let summary = "Ovewrite the contents of tensor with values from another.";
  let description = [{
    Replaces the contents of `overwritten` with corresponding values from
    `value`.

    Immediately after this op has completed, indexing `overwritten` will result
    in identical values as indexing into `tensor`. Of course, later ops
    might mutate `overwritten`, so this relationship need not hold for the
    entire program.

    This op has undefined behavior if the two tensors have different
    shapes or dtypes.
  }];
  let arguments = (ins
    AnyTorchTensorType:$value,
    AnyTorchTensorType:$overwritten
  );
  let results = (outs
  );
  let assemblyFormat = [{
    $value `overwrites` $overwritten attr-dict
      `:` type($value) `,` type($overwritten)
  }];
}

def Torch_ToBuiltinTensorOp : Torch_Op<"to_builtin_tensor", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Convert a `!torch.vtensor` to a `tensor`";
  let description = [{
    This op only operates on ValueTensorType, to avoid conflating conversions
    between value-semantic and non-value-semantic types.
  }];
  let arguments = (ins
    Torch_ValueTensorType:$operand
  );
  let results = (outs
    AnyTensor:$result
  );
  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];
}

def Torch_FromBuiltinTensorOp : Torch_Op<"from_builtin_tensor", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Convert a `tensor` to a `!torch.vtensor`";
  let description = [{
    This op only operates on ValueTensorType, to avoid conflating conversions
    between value-semantic and non-value-semantic types.
  }];
  let arguments = (ins
    AnyTensor:$operand
  );
  let results = (outs
    Torch_ValueTensorType:$result
  );
  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];
}

def Torch_ConstantNoneOp : Torch_Op<"constant.none",
    [ConstantLike, NoSideEffect]> {
  let summary = "Get the singleton None value.";
  let description = [{
    Not to be confused with the `mlir::NoneType`. Be careful to use
    `Torch::NoneType` to avoid namespace ambiguity.
  }];
  let arguments = (ins);
  let results = (outs Torch_NoneType:$result);
  let assemblyFormat = "attr-dict";
  let hasFolder = 1;
}

def Torch_ConstantStrOp : Torch_Op<"constant.str",
    [ConstantLike, NoSideEffect]> {
  let summary = "Materialize a constant str value.";
  let description = [{
    Note: Strings in Python (and TorchScript) are immutable.
  }];
  let arguments = (ins
    StrAttr:$value
  );
  let results = (outs
    Torch_StringType:$result
  );
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

#endif // TORCH_OPS
