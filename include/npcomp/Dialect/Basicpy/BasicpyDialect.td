//===- BasicPyDialect.td - Basic python dialect ------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_BASICPY_BASICPY_DIALECT
#define NPCOMP_DIALECT_BASICPY_BASICPY_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def Basicpy_Dialect : Dialect {
  let name = "basicpy";
  let summary = "Basic Python dialect";
  let description = [{
    Core types and ops 
  }];
  let cppNamespace = "Basicpy";
}

//===----------------------------------------------------------------------===//
// Op templates
//===----------------------------------------------------------------------===//

class Basicpy_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Basicpy_Dialect, mnemonic, traits> {
  let parser = [{ return parse$cppClass(parser, &result); }];
  let printer = [{ return print$cppClass(p, *this); }];  
}

//===----------------------------------------------------------------------===//
// Dialect types
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//


#endif // NPCOMP_DIALECT_BASICPY_BASICPY_DIALECT
