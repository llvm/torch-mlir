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

class RD_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<RD_Dialect, mnemonic, traits> {
}

def RD_RangeOp : RD_Op<"range"> {
    let summary = "Range of values.";
    let description = [{
        A sequence of consecutive integers.
    }];
}

#endif // #ifndef RD_OPS
