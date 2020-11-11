#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Value coders for built-in and common scenarios."""

from typing import Union

from _npcomp.mlir import ir

from .interfaces import *

__all__ = [
    "BuiltinsValueCoder",
]

_NotImplementedType = type(NotImplemented)


class BuiltinsValueCoder(ValueCoder):
  """Value coder for builtin python types."""
  __slots__ = []

  def code_py_value_as_const(self, env: Environment,
                             py_value) -> Union[_NotImplementedType, ir.Value]:
    ir_h = env.ir_h
    ir_c = ir_h.context
    if py_value is True:
      return ir_h.basicpy_bool_constant_op(True).result
    elif py_value is False:
      return ir_h.basicpy_bool_constant_op(False).result
    elif py_value is None:
      return ir_h.basicpy_singleton_op(ir_h.basicpy_NoneType).result
    elif isinstance(py_value, int):
      ir_type = env.target.impl_int_type
      ir_attr = ir_c.integer_attr(ir_type, py_value)
      return ir_h.constant_op(ir_type, ir_attr).result
    elif isinstance(py_value, float):
      ir_type = env.target.impl_float_type
      ir_attr = ir_c.float_attr(ir_type, py_value)
      return ir_h.constant_op(ir_type, ir_attr).result
    elif isinstance(py_value, str):
      return ir_h.basicpy_str_constant_op(py_value).result
    elif isinstance(py_value, bytes):
      return ir_h.basicpy_bytes_constant_op(py_value).result
    elif isinstance(py_value, type(...)):
      return ir_h.basicpy_singleton_op(ir_h.basicpy_EllipsisType).result
    return NotImplemented
