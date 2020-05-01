#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from npcomp.native.mlir.ir import *

__all__ = [
  "load_builtin_module",
]


def load_builtin_module(context = None):
  """Loads a module populated with numpy built-ins.

  This is not a long-term solution but overcomes some bootstrapping
  issues.

    >>> m = load_builtin_module()
    >>> op = m.region(0).blocks.front.operations.front
    >>> print(op.name)
    numpy.generic_ufunc

  Args:
    context: The MLIRContext to use (None to create a new one).
  Returns:
    A ModuleOp.
  """
  if context is None: context = MLIRContext()
  return context.parse_asm(_BUILTIN_MODULE_ASM)


_BUILTIN_MODULE_ASM = r"""
  numpy.generic_ufunc @numpy.add (
    // CHECK-SAME: overload(%arg0: i32, %arg1: i32) -> i32 {
    overload(%arg0: i32, %arg1: i32) -> i32 {
      // CHECK: addi
      %0 = addi %arg0, %arg1 : i32
      numpy.ufunc_return %0 : i32
    },
    // CHECK: overload(%arg0: f32, %arg1: f32) -> f32 {
    overload(%arg0: f32, %arg1: f32) -> f32 {
      // CHECK: addf
      %0 = addf %arg0, %arg1 : f32
      numpy.ufunc_return %0 : f32
    }
  )
"""

if __name__ == "__main__":
  import doctest
  doctest.testmod()
