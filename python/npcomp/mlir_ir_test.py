#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test for the MLIR IR Python bindings"""

from npcomp.native.mlir import ir
from npcomp.utils import test_utils

test_utils.start_filecheck_test()
c = ir.MLIRContext()

# CHECK-LABEL: module @parseSuccess
m = c.parse_asm(r"""
module @parseSuccess {
  func @f() {
    return
  }
}
""")
print(m.to_asm())

# CHECK-LABEL: PARSE_FAILURE
print("PARSE_FAILURE")
try:
  m = c.parse_asm("{{ILLEGAL SYNTAX}}")
except ValueError as e:
  # CHECK: [ERROR]: expected operation name in quotes
  print(e)


test_utils.end_filecheck_test(__file__)
