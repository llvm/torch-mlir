//===- NumpyOps.td - Core numpy dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_NUMPY_IR_NUMPY_OPS
#define NPCOMP_DIALECT_NUMPY_IR_NUMPY_OPS

include "NumpyDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//----------------------------------------------------------------------------//
// IR casting and conversions
//----------------------------------------------------------------------------//

def Numpy_NarrowOp : Numpy_Op<"narrow", []> {
  let summary = "Narrows an array to a known type at boundaries.";
  let description = [{
    During tracing, specific data types are often unknown. This op generically
    narrows from an unknown to a known data type at boundaries.
  }];
  let arguments = (ins
    Numpy_AnyArray:$operand
  );
  let results = (outs
    Numpy_AnyArray:$result
  );
  let assemblyFormat = [{
    $operand attr-dict `:` functional-type($operand, $result)
  }];  
}

//----------------------------------------------------------------------------//
// Universal function ops (ufunc)
// See: https://docs.scipy.org/doc/numpy/reference/ufuncs.html
//----------------------------------------------------------------------------//

def Numpy_BuiltinUfuncOp : Numpy_Op<"builtin_ufunc", [Symbol]> {
  let summary = "References a built-in universal function";
  let description = [{
    This module-level op binds by name to a fully-qualified numpy built-in
    ufunc (i.e. "numpy.add") and carries metadata associated with it.
  }];
}

def Numpy_GenericUfuncOp : Numpy_Op<"generic_ufunc", [
    IsolatedFromAbove,
    Symbol]> {
  let summary = "Defines a ufunc in terms of overloaded element-wise functions";
  let description = [{
  }];

  let arguments = (ins
    TypeArrayAttr:$overload_types);

  let regions = (region 
    VariadicRegion<AnyRegion>:$overloads);
}

def Numpy_UfuncReturnOp : Numpy_Op<"ufunc_return", [
    Terminator,
    HasParent<"Numpy::GenericUfuncOp">]> {
  let summary = "Return a value from a generic_ufunc";
  let description = [{
    Must terminate the basic block of a generic_ufunc overload.
  }];
  let arguments = (ins
    Variadic<AnyType>:$operands
  );

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Numpy_UfuncCallOp : Numpy_Op<"ufunc_call", []> {
  let summary = "Default operation on a func";
  let description = [{
    Invokes a ufunc with the given arguments. This variant models the __call__
    behavior of a python ufunc except that it does not model the `out`
    parameter, which indicates an in-place update.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$ufunc_ref,
    Variadic<Numpy_AnyArray>:$operands
  );
  let results = (outs
    Numpy_AnyArray:$result
  );

  let assemblyFormat = [{
    $ufunc_ref `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];  
}

//----------------------------------------------------------------------------//
// Built-in array functions
//
// These are ops that mirror supported array functions in numpy or related
// libraries. Note that there is some evolution happening on the dispatch
// mechanism for these.
// See: https://numpy.org/neps/nep-0018-array-function-protocol.html
// See: https://numpy.org/neps/nep-0037-array-module.html
//
// Note that operators are in general free to take any arguments, but there
// are some conventions that are mirrored here:
//
// - `out` arguments indicate that the operation should perform a mutation
//   of a specific array. This is not modeled at the individual op level,
//   instead producing IR constructs to map the intent.
//----------------------------------------------------------------------------//

def Numpy_DotOp : Numpy_Op<"dot", []> {
  let summary = "Represents the `numpy.dot` operator";
  let description = [{
    See: https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
  }];
  let arguments = (ins
    Numpy_AnyArray:$a,
    Numpy_AnyArray:$b
  );
  let results = (outs
    Numpy_AnyArray:$output
  );
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, $output)
  }];  
}

def Numpy_TransposeOp : Numpy_Op<"transpose", []> {
  let summary = "Represents the `numpy.transpose` op with no permutation specified";
  let description = [{
    This op is equivalent to calling `numpy.transpose(arr)`, which reverses
    the axes of the array. It is separate from the explicit form because it
    is not always possible to locallly infer an appropriate axis transform
    at the point of declaration.

    See: https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html
  }];
  let arguments = (ins
    Numpy_AnyArray:$a
  );
  let results = (outs
    Numpy_AnyArray:$output
  );
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, $output)
  }];  
}

//----------------------------------------------------------------------------//
// Slicing
// See: https://docs.scipy.org/doc/numpy/user/basics.indexing.html
//----------------------------------------------------------------------------//

def Numpy_GetSlice : Numpy_Op<"get_slice", []> {
  let summary = "Gets a slice of an array";
  let description = [{
    This op encapsulates all forms of indexing into an array by taking a
    variable number of `slice` arguments, each of which represents a single
    entry in a generalized indexing-tuple. Once full type inference has
    been performed, there should be sufficient static information to determine
    the exact slice semantics solely by the signature of types of the `slice`
    arguments.

    Note that there is a more general form of this op that is generally
    needed for AST extraction that takes a variable length `tuple` instead
    of a static list of arguments. It is expected that during type refinement
    most such uses should degenerate to this static variant.

    Per numpy semantics, many forms of slice return a view instead of a copy,
    and determining the exact form requires additional analysis.
  }];
  let arguments = (ins
    Numpy_AnyArray:$a,
    Variadic<Numpy_SliceTupleElement>:$slice_elements
  );
  let results = (outs
    Numpy_AnyArray:$result
  );
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, $result)
  }];  
}

#endif // NPCOMP_DIALECT_NUMPY_IR_NUMPY_OPS
