#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Value coders for built-in and common scenarios."""

from typing import Union

from .interfaces import *

from ... import ir as _ir
from ...dialects import std as std_ops, basicpy as basicpy_ops

__all__ = [
    "BuiltinsValueCoder",
]

_NotImplementedType = type(NotImplemented)


class BuiltinsValueCoder(ValueCoder):
  """Value coder for builtin python types."""
  __slots__ = []

  def code_py_value_as_const(self, env: Environment,
                             py_value) -> Union[_NotImplementedType, _ir.Value]:
    ic = env.ic
    with ic.loc, ic.ip:
      if py_value is True:
        return basicpy_ops.BoolConstantOp(ic.bool_type, ic.i1_true).result
      elif py_value is False:
        return basicpy_ops.BoolConstantOp(ic.bool_type, ic.i1_false).result
      elif py_value is None:
        return basicpy_ops.SingletonOp(ic.none_type).result
      elif isinstance(py_value, int):
        ir_type = env.target.impl_int_type
        ir_attr = _ir.IntegerAttr.get(ir_type, py_value)
        return std_ops.ConstantOp(ir_type, ir_attr).result
      elif isinstance(py_value, float):
        ir_type = env.target.impl_float_type
        ir_attr = _ir.FloatAttr.get(ir_type, py_value)
        return std_ops.ConstantOp(ir_type, ir_attr).result
      elif isinstance(py_value, str):
        return basicpy_ops.StrConstantOp(ic.str_type,
                                         _ir.StringAttr.get(py_value)).result
      elif isinstance(py_value, bytes):
        return basicpy_ops.BytesConstantOp(ic.bytes_type,
                                           _ir.StringAttr.get(py_value)).result
      elif isinstance(py_value, type(...)):
        return basicpy_ops.SingletonOp(ic.ellipsis_type).result
      return NotImplemented
