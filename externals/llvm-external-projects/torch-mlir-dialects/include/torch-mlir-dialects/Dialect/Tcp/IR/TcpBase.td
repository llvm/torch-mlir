//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_DIALECT_TCP_BASE
#define TORCH_MLIR_DIALECT_TCP_BASE

include "mlir/IR/OpBase.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

def Tcp_Dialect : Dialect {
  let name = "tcp";
  let cppNamespace = "::mlir::tcp";
  let description = [{
    Tensor Compute Primitives (TCP) dialect.

    TCP is a mid-level transformation oriented IR for deep learning & similar
    applications.
  }];

  let dependentDialects = [];
}

//===----------------------------------------------------------------------===//
// Tcp Type Definitions.
//===----------------------------------------------------------------------===//

def Tcp_Scalar : AnyTypeOf<[AnyFloat, AnySignlessInteger, AnyComplex]>;
def Tcp_Tensor : RankedTensorOf<[Tcp_Scalar]>;

def Tcp_FloatTensor : RankedTensorOf<[AnyFloat]>;
def Tcp_FloatOrComplexTensor : RankedTensorOf<[AnyFloat, AnyComplex]>;

//===----------------------------------------------------------------------===//
// Tcp Ops Base.
//===----------------------------------------------------------------------===//

class Tcp_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tcp_Dialect, mnemonic, traits> {
}

class Tcp_UnaryElementwiseOp<string mnemonic, list<Trait> traits = []> :
    Tcp_Op<mnemonic, !listconcat(traits, [
        Pure,
        Elementwise,
        SameOperandsAndResultShape,
        SameOperandsAndResultElementType])> {
}

class Tcp_BinaryElementwiseOp<string mnemonic, list<Trait> traits = []> :
    Tcp_Op<mnemonic, !listconcat(traits, [
        NoMemoryEffect,
        Elementwise,
        SameOperandsAndResultShape])> {
}

#endif // TORCH_MLIR_DIALECT_TCP_BASE
