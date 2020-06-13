#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from npcomp.compiler.frontend import *


def binary_expression():
  a = 1
  b = 100
  c = a * b + 4
  c = c * 2.0
  return c


fe = ImportFrontend()
try:
  f = fe.import_global_function(binary_expression)
finally:
  print(fe.ir_module.to_asm(debug_info=True))
