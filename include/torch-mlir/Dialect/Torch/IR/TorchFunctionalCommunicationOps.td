//===-------------------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
// Operation summaries and descriptions were systematically derived from public
// API docstrings and are licensed accordingly:
//   https://github.com/pytorch/pytorch/blob/master/LICENSE
//===----------------------------------------------------------------------===//
//
// This file is manully added because nightly-build torch's signature is
// not convergent. Generate this file if torch has stable op signature.
//
//===----------------------------------------------------------------------===//

def Torch_C10dFunctionalAllReduceOp : Torch_Op<"c10d_functional.all_reduce", [
    AllowsTypeRefinement,
    HasValueSemantics,
    ReadOnly
  ]> {
  let summary = "Generated op for `c10d_functional::all_reduce : (Tensor, str, str, int[], int) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$self,
    Torch_StringType:$reduceOp,
    Torch_StringType:$tag,
    AnyTorchListOfTorchIntType:$ranks,
    Torch_IntType:$group_size
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let hasCustomAssemblyFormat = 1;
  let extraClassDefinition = [{
    ParseResult C10dFunctionalAllReduceOp::parse(OpAsmParser &parser, OperationState &result) {
      return parseDefaultTorchOp(parser, result, 5, 1);
    }
    void C10dFunctionalAllReduceOp::print(OpAsmPrinter &printer) {
      printDefaultTorchOp(printer, *this, 5, 1);
    }
  }];
}

def Torch_C10dFunctionalWaitTensorOp : Torch_Op<"c10d_functional.wait_tensor", [
    AllowsTypeRefinement,
    HasValueSemantics,
    ReadOnly
  ]> {
  let summary = "Generated op for `c10d_functional::wait_tensor : (Tensor) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$self
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let hasCustomAssemblyFormat = 1;
  let extraClassDefinition = [{
    ParseResult C10dFunctionalWaitTensorOp::parse(OpAsmParser &parser, OperationState &result) {
      return parseDefaultTorchOp(parser, result, 1, 1);
    }
    void C10dFunctionalWaitTensorOp::print(OpAsmPrinter &printer) {
      printDefaultTorchOp(printer, *this, 1, 1);
    }
  }];
}
