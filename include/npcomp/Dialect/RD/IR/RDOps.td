//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RD_OPS
#define RD_OPS

include "npcomp/Dialect/RD/IR/RDBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class RD_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<RD_Dialect, mnemonic, traits> {
}

class RD_DatasetOp<string mnemonic>
    : RD_Op<mnemonic, [NoSideEffect]> {
}

def RD_RangeOp : RD_DatasetOp<"range"> {
    let summary = "Range of values.";
    let description = [{
        A sequence of consecutive integers.
    }];

    let arguments = (ins I64:$start, I64:$end);
    let results = (outs RD_Dataset:$result);

    let assemblyFormat = "$start `to` $end attr-dict `:` functional-type(operands, results)";
}

#endif // #ifndef RD_OPS
