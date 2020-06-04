#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# E2E demo for compiling and running various adds.

from _npcomp.mlir import ir
from _npcomp.mlir import passes

# TODO: Trace the function instead of starting from asm.
ASM = r"""
func @rank1(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "tcf.add"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
"""

context = ir.MLIRContext()
input_module = context.parse_asm(ASM)
pm = passes.PassManager(context)

pm.addPassPipelines("e2e-lowering-pipeline")
pm.run(input_module)

print(input_module.to_asm())
