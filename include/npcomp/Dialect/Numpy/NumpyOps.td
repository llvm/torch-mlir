//===- NumpyOps.td - Core numpy dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_NUMPY_NUMPY_OPS
#define NPCOMP_DIALECT_NUMPY_NUMPY_OPS

include "NumpyDialect.td"
include "mlir/Interfaces/SideEffects.td"
include "mlir/IR/SymbolInterfaces.td"

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
    HasParent<"numpy::GenericUfuncOp">]> {
  let summary = "Return a value from a generic_ufunc";
  let description = [{
    Must terminate the basic block of a generic_ufunc overload.
  }];
  let arguments = (ins
    Variadic<AnyType>:$operands
  );

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif // NPCOMP_DIALECT_NUMPY_NUMPY_OPS
