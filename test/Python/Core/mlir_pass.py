# RUN: %PYTHON %s | FileCheck %s --dump-input=fail

#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test for the MLIR Pass Python bindings"""

from _npcomp.mlir import ir
from _npcomp.mlir import passes

c = ir.MLIRContext()

pm = passes.PassManager(c)

# CHECK-LABEL: module @parseSuccess
m = c.parse_asm(r"""
module @parseSuccess {
  func @notUsed() attributes { sym_visibility = "private" }
  func @f() {
    return
  }
}
""")
# CHECK: func private @notUsed
# CHECK: func @f
print(m.to_asm())

# CHECK: PASS COUNT: 0
print("PASS COUNT:", len(pm))

pm.addPassPipelines("canonicalize", "symbol-dce")
# Note: not checking the actual count since these may expand to more than
# two passes.
# CHECK: PASS COUNT:
print("PASS COUNT:", len(pm))
# CHECK: PASSES: canonicalize, symbol-dce
print("PASSES:", str(pm))
pm.run(m)
print(m.to_asm())
# CHECK-NOT: func @notUsed
