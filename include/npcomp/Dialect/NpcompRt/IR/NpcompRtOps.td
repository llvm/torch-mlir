//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMPRT_OPS
#define NPCOMPRT_OPS

include "npcomp/Dialect/NpcompRt/IR/NpcompRtBase.td"

class NpcompRt_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<NpcompRt_Dialect, mnemonic, traits> {
}

def NpcompRt_GetExtentOp : NpcompRt_Op<"get_extent"> {
  let summary = "Gets the specified extent of a buffer view.";
  let description = [{
    Gets the specified extent of a buffer view.
  }];
  let arguments = (ins NpcompRt_BufferView:$bufferView, I64Attr:$dim);
  let results = (outs Index:$extent);
}

#endif // #ifndef NPCOMPRT_OPS
