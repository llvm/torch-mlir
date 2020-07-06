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
include "npcomp/Typing/Analysis/CPA/Interfaces.td"
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
// NdArray type handling
//----------------------------------------------------------------------------//

def Numpy_CreateArrayFromTensorOp : Numpy_Op<"create_array_from_tensor", [
    DeclareOpInterfaceMethods<NPCOMP_CPATypeInferenceOpInterface>,
    NoSideEffect]> {
  let summary = "Creates an ndarray from a tensor.";
  let description = [{
    Creates a new ndarray that will contain the data of the given tensor.
  }];
  let arguments = (ins
    Numpy_AnyTensor:$source
  );
  let results = (outs
    Numpy_AnyArray:$dest
  );
  let assemblyFormat = [{
    $source attr-dict `:` functional-type($source, $dest)
  }];
}

def Numpy_CopyToTensorOp : Numpy_Op<"copy_to_tensor", [
    DeclareOpInterfaceMethods<NPCOMP_CPATypeInferenceOpInterface>]> {
  let summary = "Copies an ndarray, yielding a value-typed tensor.";
  let description = [{
    The semantics of this operation connote a copy of the data in the source
    ndarray, producing a destination value that will have the value in the
    ndarray at the point of the copy. Of course, downstream transformations
    are free to rearrange things to elide the copy or otherwise eliminate the
    need for it.
  }];
  let arguments = (ins
    Numpy_NdArrayType:$source
  );
  let results = (outs
    Numpy_AnyTensor:$dest
  );
  let assemblyFormat = [{
    $source attr-dict `:` functional-type($source, $dest)
  }];
  let hasCanonicalizer = 1;
}

//----------------------------------------------------------------------------//
// Universal function ops (ufunc)
// See: https://docs.scipy.org/doc/numpy/reference/ufuncs.html
//----------------------------------------------------------------------------//

def Numpy_BuiltinUfuncCallOp : Numpy_Op<"builtin_ufunc_call", [
    DeclareOpInterfaceMethods<NPCOMP_CPATypeInferenceOpInterface>]> {
  let summary = "A __call__ operation on a named/builtin ufunc";
  let description = [{
    Simple ufunc call semantics for builtin ufuncs with none of the advanced
    arguments specified.

    Note that without the `out=` parameter, ufunc call operations (unlike
    others like `at`) are defined purely in the value domain and do not alias.
    As such, they operate on tensors, not ndarray.
  }];
  let arguments = (ins
    StrAttr:$qualified_name,
    Variadic<Numpy_AnyTensor>:$inputs
  );
  let results = (outs
    Numpy_AnyTensor:$output
  );
  let assemblyFormat = [{
    `<` $qualified_name `>` `(` operands `)` attr-dict `:` functional-type(operands, results)
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
