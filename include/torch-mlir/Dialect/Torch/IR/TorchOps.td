//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_OPS
#define TORCH_OPS

include "torch-mlir/Dialect/Torch/IR/TorchTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Torch_Op<string mnemonic, list<Trait> traits = []>
    : Op<Torch_Dialect, mnemonic, traits> {
}

include "torch-mlir/Dialect/Torch/IR/GeneratedAtenOps.td"
include "torch-mlir/Dialect/Torch/IR/GeneratedPrimOps.td"
include "torch-mlir/Dialect/Torch/IR/GeneratedQuantizedOps.td"

//===----------------------------------------------------------------------===//
// TorchScript `torch.nn.Module` object instantiation ops.
//===----------------------------------------------------------------------===//

def Torch_NnModuleOp : Torch_Op<"nn_module", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    SingleBlockImplicitTerminator<"::mlir::torch::Torch::NnModuleTerminatorOp">]> {
  let summary = "Constructs a torch.nn.Module";
  let description = [{
    This op is used to represent a torch.nn.Module when importing a
    graph of Python objects.

    This op returns a new torch.nn.Module as an SSA value, with a set of
    declaratively specified properties.

    Example:

    ```mlir
    %2 = torch.nn_module {
      torch.slot "b", %bool_true : !torch.bool
      torch.slot "i", %int3 : !torch.int
      torch.slot "f", %float : !torch.float
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
  let hasVerifier = 1;

  let assemblyFormat = "$region attr-dict `:` qualified(type($result))";

  let extraClassDeclaration = [{
    StringRef getClassName() { return getType().getClassName(); }
    ClassTypeOp getClassType(::mlir::SymbolTable &symbolTable) {
      return symbolTable.lookup<ClassTypeOp>(getClassName());
    }
  }];
}

def Torch_NnModuleTerminatorOp : Torch_Op<"nn_module_terminator", [Terminator,
    HasParent<"::mlir::torch::Torch::NnModuleOp">]> {
  let summary = "Implicit terminator for torch.nn_module";

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "attr-dict";
}

def Torch_SlotOp : Torch_Op<"slot", [
    HasParent<"::mlir::torch::Torch::NnModuleOp">]> {
  let summary = "Define the value of a slot of a torch.nn.Module";
  let description = [{
    This op specifies that the initial value of the slot `name` of the
    parent torch.nn_module should be `value`, which is allowed to be an
    arbitrary Torch-compatible SSA value, including other !torch.nn.Module's.
  }];

  let arguments = (ins StrAttr:$name, AnyTorchType:$value);
  let results = (outs);

  let assemblyFormat = [{
    $name `,` $value attr-dict `:` qualified(type($value))
  }];
}

//===----------------------------------------------------------------------===//
// Modeling of TorchScript class types
//===----------------------------------------------------------------------===//

def Torch_ClassTypeOp : Torch_Op<"class_type", [
    Symbol,
    SingleBlockImplicitTerminator<"::mlir::torch::Torch::ClassTypeTerminatorOp">]> {
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
      torch.attr "b" : !torch.bool
      torch.attr "i" : !torch.int
      torch.attr "f" : !torch.float
      torch.attr "t" : !torch.tensor
      torch.attr "submodule" : !torch.nn.Module<"empty">
      torch.method "method", @f
    }
    torch.nn_module {
      // These must match the order and names in the `torch.class_type`.
      torch.slot "b", %bool_true : !torch.bool
      torch.slot "i", %int3 : !torch.int
      torch.slot "f", %float : !torch.float
      torch.slot "t", %t : !torch.tensor
      torch.slot "submodule", %submodule : !torch.nn.Module<"empty">
    } : !torch.nn.Module<"test">
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let results = (outs);
  let regions = (region SizedRegion<1>:$region);
  let hasVerifier = 1;

  let assemblyFormat = "$sym_name $region attr-dict";
}

def Torch_ClassTypeTerminatorOp : Torch_Op<"class_type_terminator", [Terminator,
    HasParent<"::mlir::torch::Torch::ClassTypeOp">]> {
  let summary = "Implicit terminator for torch.class_type";

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "attr-dict";
}

def Torch_MethodOp : Torch_Op<"method", [
    HasParent<"::mlir::torch::Torch::ClassTypeOp">,
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
    HasParent<"::mlir::torch::Torch::ClassTypeOp">
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
    SingleBlockImplicitTerminator<"::mlir::torch::Torch::GlobalSlotInitOp">
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
    HasParent<"::mlir::torch::Torch::GlobalSlotOp">]> {
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

  let assemblyFormat = "$initialValue attr-dict `:` qualified(type($initialValue))";
}

def Torch_GlobalSlotGetOp : Torch_Op<"global_slot.get", []> {
  let summary = "Get the value stored in a torch.global_slot";

  let arguments = (ins
    FlatSymbolRefAttr:$slot
  );
  let results = (outs AnyTorchType:$result);

  let assemblyFormat = [{
    $slot attr-dict `:` qualified(type($result))
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
    $slot `=` $value attr-dict `:` qualified(type($value))
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
    $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($results))
  }];
}

def Torch_PrimTupleConstructOp: Torch_Op<"prim.TupleConstruct", [
    NoSideEffect,
    TypesMatchWith<"contained types correspond to operand types",
    "elements", "result", "Torch::TupleType::get($_ctxt, llvm::to_vector<6>($_self))",
    "isValidSubtype">
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
    $elements attr-dict `:` qualified(type($elements)) `->` qualified(type($result))
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

  let hasVerifier = 1;

  let assemblyFormat = [{
    $elements attr-dict `:` functional-type(operands, results)
  }];
}

def Torch_PrimDictConstructOp: Torch_Op<"prim.DictConstruct", [
    AllowsTypeRefinement,
    SameVariadicOperandSize,
    ]> {
  let summary = "TorchScript prim::DictConstruct op";

  let arguments = (ins
    Variadic<AnyTorchDictKeyType>:$keys,
    Variadic<AnyTorchType>:$values
  );

  let results = (outs
    Torch_DictType:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `keys` `(` ($keys^ `:` qualified(type($keys)))? `)` `values` `(` ($values^ `:` qualified(type($values)))? `)` attr-dict `->` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    Type getKeyType() { return getType().cast<DictType>().getKeyType(); }
    Type getValueType() { return getType().cast<DictType>().getValueType(); }
  }];
}

def Torch_PrimCreateObjectOp: Torch_Op<"prim.CreateObject", [
    NoSideEffect,
    AllowsTypeRefinement,
    ]> {
  let summary = "TorchScript prim::CreateObject op";

  let arguments = (ins
  );
  let results = (outs
    Torch_NnModuleType:$result
  );

  let assemblyFormat = [{
    attr-dict qualified(type($result))
  }];
}

def Torch_PrimGetAttrOp : Torch_Op<"prim.GetAttr", []> {
  let summary = "TorchScript prim::GetAttr op";

  let arguments = (ins StrAttr:$name, Torch_NnModuleType:$receiver);
  let results = (outs AnyTorchType:$result);

  let assemblyFormat = [{
    $receiver `[` $name `]` attr-dict `:` qualified(type($receiver)) `->` qualified(type($result))
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
    $receiver `[` $name `]` `=` $value attr-dict `:` qualified(type($receiver)) `,` qualified(type($value))
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
    $receiver `[` $name `]` `(` $operands `)` attr-dict `:` qualified(type($receiver)) `,` functional-type($operands, $result)
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
    Torch_IntType:$maxTripCount,
    Torch_BoolType:$initialCondition,
    Variadic<AnyTorchType>:$iterArgsInit
  );
  let results = (outs Variadic<AnyTorchType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = [{
    $maxTripCount `,` $initialCondition `,` `init` `(` $iterArgsInit `)` $region
    attr-dict `:` functional-type(operands, results)
  }];
  let extraClassDeclaration = [{
    /// Returns true if this loop is "for-like". Otherwise it is "while-like"
    /// and this function returns false.
    /// See https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md#loops
    bool isForLike();
  }];
}

def Torch_PrimLoopConditionOp : Torch_Op<"prim.Loop.condition", [
    DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>,
    Terminator,
    HasParent<"::mlir::torch::Torch::PrimLoopOp">]> {
  let summary = "yield-like terminator for torch.prim.Loop";
  let description = [{
    Does not correspond to any torch prim op directly (the way that they model
    blocks has a built-in notion of yield-like terminator).
  }];

  let arguments = (ins
    Torch_BoolType:$shouldContinue,
    Variadic<AnyTorchType>:$iterArgs
  );
  let results = (outs);

  let assemblyFormat = [{
    $shouldContinue `,`
    `iter` `(` ($iterArgs^ `:` qualified(type($iterArgs)))? `)` attr-dict
  }];
}

def Torch_PrimIfOp : Torch_Op<"prim.If", [
  DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "TorchScript prim::If op";
  let description = [{
    This op (together with prim.If.yield) define a conditional control flow
    construct. It is analogous to `scf.if` for MLIR folks that are familiar
    with that. The main differences from that op are:

    - `!torch.bool` condition value.
    - The "else" region is always present. This is reflective of invariants of
      the TorchScript IR.
    - No special prettiness for the "no yielded values" case. These are
      interesting for modeling mostly-non-SSA programs, but TorchScript IR
      is already in SSA form.

    See: https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md#if
  }];

  let arguments = (ins Torch_BoolType:$condition);
  let results = (outs Variadic<AnyTorchType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion, SizedRegion<1>:$elseRegion);
  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizer = 1;
}

def Torch_PrimIfYieldOp : Torch_Op<"prim.If.yield", [
    Terminator,
    ReturnLike,
    NoSideEffect,
    HasParent<"::mlir::torch::Torch::PrimIfOp">]> {
  let summary = "yield-like terminator for torch.prim.If";
  let description = [{
    Does not correspond to any torch prim op directly (the way that they model
    blocks has a built-in notion of yield-like terminator).
  }];

  let arguments = (ins
    Variadic<AnyTorchType>:$results
  );
  let results = (outs);

  let assemblyFormat = [{
    attr-dict ($results^ `:` qualified(type($results)))?
  }];
}

//===----------------------------------------------------------------------===//
// Ops corresponding to prim::Constant
//===----------------------------------------------------------------------===//

def Torch_ConstantNoneOp : Torch_Op<"constant.none",
    [ConstantLike, NoSideEffect,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
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
    [ConstantLike, NoSideEffect,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
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

def Torch_ConstantDeviceOp : Torch_Op<"constant.device",
    [NoSideEffect,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Materialize a constant Device value.";
  let description = [{

  }];
  let arguments = (ins
    StrAttr:$value
  );
  let results = (outs
    Torch_DeviceType:$result
  );
  let assemblyFormat = "$value attr-dict";
}

def Torch_ConstantIntOp : Torch_Op<"constant.int",
    [ConstantLike, NoSideEffect,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Materialize a constant `int` value.";
  let description = [{
    Note: TorchScript represents integers as 64-bit signed values, unlike
    Python where they are arbitrary precision.
  }];
  let arguments = (ins
    AnyI64Attr:$value
  );
  let results = (outs
    Torch_IntType:$result
  );
  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
}

def Torch_ConstantFloatOp : Torch_Op<"constant.float",
    [ConstantLike, NoSideEffect,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Materialize a constant `float` value.";
  let description = [{
    Note: TorchScript represents `float` as 64-bit floating point values.

    TODO: Add a `!torch.float` type.
  }];
  let arguments = (ins
    F64Attr:$value
  );
  let results = (outs
    Torch_FloatType:$result
  );
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

def Torch_ConstantBoolOp : Torch_Op<"constant.bool",
    [ConstantLike, NoSideEffect,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Materialize a constant `bool` value.";
  let description = [{
  }];
  let arguments = (ins
    I1Attr:$value
  );
  let results = (outs
    Torch_BoolType:$result
  );
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
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
    $operand attr-dict `:` qualified(type($operand)) `to` qualified(type($result))
  }];

  let hasFolder = 1;
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
    $weight (`,` $bias^)? attr-dict `:` qualified(type($weight)) (`,` qualified(type($bias))^)?
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
    Torch_FloatType:$scale,
    Torch_IntType:$offset
  );
  // TODO: Limit to quantized dtypes (e.g. !torch.qint8).
  let results = (outs AnyTorchTensorType:$result);

  let assemblyFormat = [{
    $int_repr `,` $scale `,` $offset attr-dict
    `:` qualified(type($int_repr)) `,` qualified(type($scale)) `,` qualified(type($offset)) `->` qualified(type($result))
  }];
}

def Torch_NonValueTensorLiteralOp : Torch_Op<"tensor.literal", [
    DeclareOpInterfaceMethods<InferTypeOpInterface, ["isCompatibleReturnTypes"]>,
    AllowsTypeRefinement,
  ]> {
  let summary = "Create a value of !torch.tensor type from a literal";
  let description = [{
    Example:
    ```
    %0 = torch.tensor.literal(dense<0.0> : tensor<3x5xf32>) : !torch.tensor
    %1 = torch.tensor.literal(dense<0.0> : tensor<3xf32>) : !torch.tensor<[3],f32>
    ```

    This op covers a typical frontend use case of creating a type-erased
    `!torch.tensor`. Inside the compiler, we decompose it into
    `torch.vtensor.literal` which is easier to analyze and transform.

    Note: This op is not called "constant" because the created tensor is not
    "constant" in any meaning of that word.
  }];
  let arguments = (ins ElementsAttr:$value);
  let results = (outs Torch_NonValueTensorType:$result);

  let assemblyFormat = [{
    `(` $value `)` attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    // InferTypeOpInterface:
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual);
  }];
}

def Torch_ValueTensorLiteralOp : Torch_Op<"vtensor.literal", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    ConstantLike,
    NoSideEffect,
  ]> {
  let summary = "Create a value of !torch.vtensor type from a literal";
  let description = [{
    Example:
    ```
    %0 = torch.vtensor.literal(dense<0.0> : tensor<3x5xf32>) : !torch.vtensor<[3,5],f32>
    %1 = torch.vtensor.literal(dense<0.0> : tensor<3xf32>) : !torch.vtensor<[3],f32>
    ```

    Unlike `torch.tensor.literal`, which covers a typical frontend use case
    and allows type refinement, this op always has a maximally resolved type
    (which is always possible, because it is created from a literal). This
    has a stronger set of invariants that better fit the needs of the
    compiler internals.
  }];
  let arguments = (ins ElementsAttr:$value);
  let results = (outs Torch_ValueTensorType:$result);

  let assemblyFormat = [{
    `(` $value `)` attr-dict `:` qualified(type($result))
  }];

  let hasFolder = 1;
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
    use `torch.copy.to_tensor` and `torch.copy.from_tensor`. This op is kept
    separate to prevent canonicalizations from accidentally dropping static
    information. In most cases, after running the `torch-refine-types` pass,
    this op becomes a no-op (the pass will incorporate the static information
    into other ops that allow type refinement).
  }];
  let arguments = (ins
    AnyTorchTensorType:$operand
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let assemblyFormat = [{
    $operand attr-dict `:` qualified(type($operand)) `to` qualified(type($result))
  }];
  let hasCanonicalizer = 1;
}

def Torch_CopyToNonValueTensorOp : Torch_Op<"copy.to_tensor", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    TypesMatchWith<"operand is corresponding !torch.vtensor",
                   "result", "operand",
                   "$_self.cast<NonValueTensorType>().getWithValueSemantics()">,
  ]> {
  let summary = "Create a !torch.tensor with the same contents as the operand";
  let description = [{
    This op is used to convert from !torch.vtensor to !torch.tensor.
    It does so by allocating a new !torch.tensor and filling it with
    the contents of the operand.

    However, this op *does not* allow adding/removing static information about
    sizes/dtype. For that, use `torch.tensor_static_info_cast`.

    This op does not have the AllowsTypeRefinement trait because the operand
    and result types are coupled. Only places that know how to simultaneously
    update both types should be changing the type of this op.
  }];
  let arguments = (ins
    Torch_ValueTensorType:$operand
  );
  let results = (outs
    Torch_NonValueTensorType:$result
  );
  let assemblyFormat = [{
    $operand attr-dict `:` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Torch_CopyToValueTensorOp : Torch_Op<"copy.to_vtensor", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    TypesMatchWith<"operand is corresponding !torch.tensor",
                   "result", "operand",
                   "$_self.cast<ValueTensorType>().getWithoutValueSemantics()">,
  ]> {
  let summary = "Create a !torch.vtensor with the same contents as the operand";
  let description = [{
    This op is used to convert from !torch.tensor to !torch.vtensor.

    However, this op *does not* allow adding/removing static information about
    sizes/dtype. For that, use `torch.tensor_static_info_cast`.

    This op does not have the AllowsTypeRefinement trait because the operand
    and result types are coupled. Only places that know how to simultaneously
    update both types should be changing the type of this op.
  }];
  let arguments = (ins
    Torch_NonValueTensorType:$operand
  );
  let results = (outs
    Torch_ValueTensorType:$result
  );
  let assemblyFormat = [{
    $operand attr-dict `:` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Torch_OverwriteTensorContentsOp : Torch_Op<"overwrite.tensor.contents", [
    TypesMatchWith<"overwritten tensor type is corresponding !torch.tensor of value tensor type",
                   "value", "overwritten",
                   "$_self.cast<ValueTensorType>().getWithoutValueSemantics()">
  ]> {
  let summary = "Ovewrite the contents of tensor with values from another.";
  let description = [{
    Replaces the contents of `overwritten` with corresponding values from
    `value`.

    Immediately after this op has completed, indexing `overwritten` will result
    in identical values as indexing into `value`. Of course, later ops
    might mutate `overwritten`, so this relationship need not hold for the
    entire program. This op only updates the tensor data (not metadata).
    In other words, it cannot change the (dynamic) shape of the overwritten tensor.

    This op does not have the AllowsTypeRefinement trait because the types of the
    two operands are coupled. Only places that know how to simultaneously update
    both types should be changing the type of this op.
  }];
  let arguments = (ins
    Torch_ValueTensorType:$value,
    Torch_NonValueTensorType:$overwritten
  );
  let results = (outs
  );
  let assemblyFormat = [{
    $value `overwrites` $overwritten attr-dict
      `:` qualified(type($value)) `,` qualified(type($overwritten))
  }];
}

// The corresponding without underscore variant for `torch.aten.uniform_`
// doesn't exist in the pytorch ops registry. Add it here.
def Torch_ValsemVariantAtenUniformOp: Torch_Op<"valsem.aten.uniform", [
    AllowsTypeRefinement,
    HasValueSemantics,
    ReadOnly
  ]> {
  let summary = "`uniform op : (Tensor, float, float, Generator?) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$self,
    Torch_FloatType:$from,
    Torch_FloatType:$to,
    TorchOptionalGeneratorType:$generator
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let assemblyFormat = "$self `,` $from `,` $to `,` $generator attr-dict `:` type($self) `,` type($from) `,` type($to) `,` type($generator) `->` type($result)";
}

// The corresponding without underscore variant for `torch.aten.bernoulli_.float`
// doesn't exist in the pytorch ops registry. Add it here.
def Torch_ValsemVariantAtenBernoulliFloatOp: Torch_Op<"valsem.aten.bernoulli.float", [
    AllowsTypeRefinement,
    HasValueSemantics,
    ReadOnly
  ]> {
  let summary = "`bernoulli.float op : (Tensor, float, Generator?) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$self,
    Torch_FloatType:$p,
    TorchOptionalGeneratorType:$generator
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let assemblyFormat = "$self `,` $p `,` $generator attr-dict `:` type($self) `,` type($p) `,` type($generator) `->` type($result)";
}

// The corresponding without underscore variant for `torch.aten.bernoulli_.Tensor`
// doesn't exist in the pytorch ops registry. Add it here.
def Torch_ValsemVariantAtenBernoulliTensorOp: Torch_Op<"valsem.aten.bernoulli.Tensor", [
    AllowsTypeRefinement,
    HasValueSemantics,
    ReadOnly
  ]> {
  let summary = "Generated op for `aten::bernoulli_.Tensor : (Tensor, Tensor, Generator?) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$self,
    AnyTorchTensorType:$p,
    TorchOptionalGeneratorType:$generator
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let assemblyFormat = "$self `,` $p `,` $generator attr-dict `:` qualified(type($self)) `,` qualified(type($p)) `,` qualified(type($generator)) `->` qualified(type($result))";
}

// The corresponding without underscore variant for `torch.aten.fill_.Scalar`
// doesn't exist in the pytorch ops registry. Add it here.
def Torch_ValsemVariantAtenFillScalarOp: Torch_Op<"valsem.aten.fill.Scalar", [
    AllowsTypeRefinement,
    HasValueSemantics,
    ReadOnly
  ]> {
  let summary = "`fill.Scalar op : (Tensor, Scalar) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$self,
    AnyTorchScalarType:$value
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let assemblyFormat = "$self `,` $value attr-dict `:` qualified(type($self)) `,` qualified(type($value)) `->` qualified(type($result))";
}

// The corresponding without underscore variant for `torch.aten._index_put_impl_`
// doesn't exist in the pytorch ops registry. Add it here.
def Torch_ValsemVariantAtenIndexPutImplOp: Torch_Op<"valsem.aten.index_put_impl", [
    AllowsTypeRefinement,
    HasValueSemantics,
    ReadOnly
  ]> {
  let summary = "`index_put_impl op : (Tensor, Tensor?[], Tensor, bool, bool) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$self,
    AnyTorchOptionalTensorListType:$indices,
    AnyTorchTensorType:$values,
    Torch_BoolType:$accumulate,
    Torch_BoolType:$unsafe
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let assemblyFormat = "$self `,` $indices `,` $values `,` $accumulate `,` $unsafe attr-dict `:` qualified(type($self)) `,` qualified(type($indices)) `,` qualified(type($values)) `,` qualified(type($accumulate)) `,` qualified(type($unsafe)) `->` qualified(type($result))";
}

// The corresponding without underscore variant for `torch.aten.copy_`
// doesn't exist in the pytorch ops registry. Add it here.
def Torch_ValsemVariantAtenCopyOp : Torch_Op<"valsem.aten.copy", [
    AllowsTypeRefinement,
    HasValueSemantics,
    ReadOnly
  ]> {
  let summary = "copy op : (Tensor, Tensor, bool) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$self,
    AnyTorchTensorType:$src,
    Torch_BoolType:$non_blocking
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let assemblyFormat = "$self `,` $src `,` $non_blocking attr-dict `:` qualified(type($self)) `,` qualified(type($src)) `,` qualified(type($non_blocking)) `->` qualified(type($result))";
}

// To handle runtime assertions, torchscript provides us `torch._assert` operation. 
// But TS compiler introduces control flow for `torch._assert` operation. The 
// `torch._assert` would introduce control flow like: 
// 
//            %cond = "torch.aten.Bool.Tensor"(%0) : (!torch.tensor) -> !torch.bool
//            "torch.prim.If"(%cond) ({
//              "torch.prim.If.yield"() : () -> ()
//            }, {
//              "torch.prim.RaiseException"(%msg) : (!torch.str) -> ()
//              "torch.prim.If.yield"() : () -> ()
//            }) : (!torch.bool) -> ()
//
// This new operation `torch.runtime.assert` is added to simplify the IR control
// flow by avoiding unnecessary branches. It also makes insertion of the runtime
// assert in the source code easier.
def Torch_RuntimeAssertOp: Torch_Op<"runtime.assert", [
    AllowsTypeRefinement,
    HasValueSemantics,
    ReadOnly
  ]> {
  let summary = "Runtime Assertion";
  let arguments = (ins
    Torch_BoolType:$condition,
    StrAttr:$message
  );
  let results = (outs
  );
  let assemblyFormat = "$condition `,` $message attr-dict";
}

//===----------------------------------------------------------------------===//
// Shape modeling ops.
//===----------------------------------------------------------------------===//

def Torch_ShapeCalculateOp : Torch_Op<"shape.calculate", [
  DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "Shape calculation encapsulation op";
  let description = [{
    The `torch.shape.calculate` op captures a shape calculation
    (in the region `shapeCalculation`) which calculates the shapes for
    the set of values yielded by the `body` region.

    The `shapeCalculation` region yields a `!torch.list<int>` for each
    value yielded by the `body` region.

    Conceptually, the `shapeCalculation` region executes first, then `body`
    region. So the `shapeCalculation` region can also contain arbitrary
    assertions or side-effecting code which guard the validity of the execution
    of the body (typically by terminating the program with a
    torch.prim.RaiseException op).

    The program has undefined behavior if the values yielded by the `body`
    region do not have the shapes yielded by the `shapeCalculation` region.
  }];

  let arguments = (ins);
  let results = (outs Variadic<AnyTorchType>:$results);
  let regions = (region
    SizedRegion<1>:$body,
    SizedRegion<1>:$shapeCalculation
  );

  let assemblyFormat = [{
    $body `shapes` $shapeCalculation attr-dict `:` type($results)
  }];
}

def Torch_ShapeCalculateYieldOp : Torch_Op<"shape.calculate.yield", [
    Terminator,
    ReturnLike,
    HasParent<"::mlir::torch::Torch::ShapeCalculateOp">]> {
  let summary = "yield-like terminator for torch.shape.calculate";
  let description = [{
    This op terminates the `body` region of a `torch.shape.calculate` op.
  }];

  let arguments = (ins
    Variadic<AnyTorchType>:$results
  );
  let results = (outs);

  let assemblyFormat = [{
    attr-dict ($results^ `:` type($results))?
  }];
}

def Torch_ShapeCalculateYieldShapesOp : Torch_Op<"shape.calculate.yield.shapes", [
    DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>,
    Terminator,
    HasValueSemantics,
    ReadOnly,
    HasParent<"::mlir::torch::Torch::ShapeCalculateOp">]> {
  let summary = "yield-like terminator for torch.shape.calculate shape region";
  let description = [{
    This op terminates the `shapeCalculation` region of a
    `torch.shape.calculate` op.
  }];

  let arguments = (ins
    Variadic<TorchIntListType>:$results
  );
  let results = (outs);

  let assemblyFormat = [{
    attr-dict ($results^ `:` type($results))?
  }];

  let hasVerifier = 1;
}

#endif // TORCH_OPS
