//===- NumpyDialect.td - Core numpy dialect ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_NUMPY_IR_NUMPY_DIALECT
#define NPCOMP_DIALECT_NUMPY_IR_NUMPY_DIALECT

include "mlir/IR/OpBase.td"
include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def Numpy_Dialect : Dialect {
  let name = "numpy";
  let summary = "Core numpy dialect";
  let description = [{
    Dialect of types and core numpy ops and abstractions.
  }];
  let cppNamespace = "Numpy";
}

//===----------------------------------------------------------------------===//
// Op templates
//===----------------------------------------------------------------------===//

class Numpy_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Numpy_Dialect, mnemonic, traits> {
  let parser = [{ return parse$cppClass(parser, &result); }];
  let printer = [{ return print$cppClass(p, *this); }];  
}

//===----------------------------------------------------------------------===//
// Dialect types
//===----------------------------------------------------------------------===//

def Numpy_AnyDtype : DialectType<Numpy_Dialect,
    CPred<"$_self.isa<::mlir::NPCOMP::Numpy::AnyDtypeType>()">, "any dtype">,
    BuildableType<"$_builder.getType<::mlir::NPCOMP::Numpy::AnyDtypeType>()"> {
  let typeDescription = [{
    Placeholder for an unknown dtype in a tensor.
  }];
}

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

// Any type, at any stage of analysis that can represent a numpy array.
def Numpy_AnyArray : TensorOf<[AnyType]>;

def Numpy_SliceTupleElement : AnyTypeOf<[
  // Supports both "Index Arrays" and "Boolean mask index arrays".
  Numpy_AnyArray,
  
  // Indicates that an axis should be added (np.newaxis == None).
  Basicpy_NoneType,

  // Indicates that intervening axes should be preserved.
  Basicpy_EllipsisType,

  // A discrete numeric index (represented as IndexType so that a proper
  // width can be target dependent).
  Index,

  // A generalized slice object.
  Basicpy_SliceSlotObjectType,
], "types that are legal elements of a __getitem__ tuple operating on arrays">;

#endif // NPCOMP_DIALECT_NUMPY_IR_NUMPY_DIALECT
