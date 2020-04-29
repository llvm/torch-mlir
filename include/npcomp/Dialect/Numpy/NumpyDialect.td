//===- NumpyDialect.td - Core numpy dialect ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_NUMPY_NUMPY_DIALECT
#define NPCOMP_DIALECT_NUMPY_NUMPY_DIALECT

include "mlir/IR/OpBase.td"

def Numpy_Dialect : Dialect {
  let name = "numpy";
  let summary = "Core numpy dialect";
  let description = [{
    Dialect of types and core numpy ops and abstractions.
  }];
  let cppNamespace = "NUMPY";
}

class Numpy_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Numpy_Dialect, mnemonic, traits> {
  let parser = [{ return parse$cppClass(parser, &result); }];
  let printer = [{ return print$cppClass(p, *this); }];  
}

#endif // NPCOMP_DIALECT_NUMPY_NUMPY_DIALECT
