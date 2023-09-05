//===- TestTorchDialectExtension.td --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operations that are injected into the Torch
// dialect through the extension mechanism, as a test.
//
//===----------------------------------------------------------------------===//

#ifndef TESTTORCHDIALECTEXTENSION_TD
#define TESTTORCHDIALECTEXTENSION_TD

include "torch-mlir/Dialect/Torch/IR/TorchBase.td"
include "torch-mlir/Dialect/Torch/IR/TorchTypes.td"

class Torch_Op<string mnemonic, list<Trait> traits = []>
    : Op<Torch_Dialect, mnemonic, traits> {
}

def Torch_GoofyIdentityOp : Torch_Op<"goofy.identity", [
    AllowsTypeRefinement,
    HasValueSemantics,
    ReadOnly
  ]> {
  let summary = "Generated op for `goofy.identity`";
  let arguments = (ins
    AnyTorchTensorType:$self
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let hasCustomAssemblyFormat = 1;
  let extraClassDefinition = [{
    ParseResult GoofyIdentityOp::parse(OpAsmParser &parser, OperationState &result) {
      return torch::Torch::parseDefaultTorchOp(parser, result, 1, 1);
    }
    void GoofyIdentityOp::print(OpAsmPrinter &printer) {
      torch::Torch::printDefaultTorchOp(printer, *this, 1, 1);
    }
  }];
  let cppNamespace = "::mlir::test";
}


#endif // TESTTORCHDIALECTEXTENSION_TD
