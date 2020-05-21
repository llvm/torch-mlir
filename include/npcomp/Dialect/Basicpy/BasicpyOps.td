//===- BasicPyOps.td - Basic Python ops --------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_BASICPY_BASICPY_OPS
#define NPCOMP_DIALECT_BASICPY_BASICPY_OPS

include "BasicpyDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def Basicpy_SingletonOp : Basicpy_Op<"singleton", [
    ConstantLike, NoSideEffect]> {
  let summary = "Constant value for a singleton type";
  let description = [{
    Some types only have a single possible value, represented by the
    SingletonAttr. This op allows creating constants of these types.
  }];
  let arguments = (ins);
  let results = (outs
    Basicpy_SingletonType:$result
  );
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Basicpy_SlotObjectMakeOp : Basicpy_Op<"slot_object_make", [
    NoSideEffect]> {
  let summary = "Creates an instance of a SlotObject type";
  let description = [{
    SlotObjects are typically instances of built-in classes that have a fixed
    number of slots. Unlike in standard python, the types of each slot are
    tracked.

    This op has a custom assembly form which can be used when valid that
    omits the operand types (since they are equal to the types in the returned
    slot object). Example:
      %0 = basicpy.singleton : !basicpy.NoneType
      %1 = basicpy.slot_object_make(%0) -> 
          !basicpy.SlotObject<slice, !basicpy.NoneType>
  }];
  let arguments = (ins
    StrAttr:$className,
    // TODO: Tighter constraints on allowable types.
    Variadic<AnyType>:$slots
  );
  let results = (outs
    Basicpy_SlotObjectType:$result
  );
}

def Basicpy_SlotObjectGetOp : Basicpy_Op<"slot_object_get", [
    NoSideEffect]> {
  let summary = "Gets a slot from a slot object";
  let description = [{
    Gets a slot from a SlotObject.

    Example:
      %0 = basicpy.slot_object_make ...
      %1 = basicpy.slot_object_get %0[1] : !basicpy.SlotObject<...>
  }];
  let arguments = (ins
    Basicpy_SlotObjectType:$object,
    IndexAttr:$index
  );
  let results = (outs
    AnyType:$result
  );
}

#endif // NPCOMP_DIALECT_BASICPY_BASICPY_OPS
