#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Value coders for Numpy types."""

import numpy as np
from typing import Union

from _npcomp.mlir import ir

from . import logging
from .interfaces import *

__all__ = [
    "CreateNumpyValueCoder",
]

_NotImplementedType = type(NotImplemented)


class NdArrayValueCoder(ValueCoder):
  """Value coder for numpy types."""
  __slots__ = []

  def code_py_value_as_const(self, env: Environment,
                             py_value) -> Union[_NotImplementedType, ir.Value]:
    # TODO: Query for ndarray compat (for duck typed and such)
    # TODO: Have a higher level name resolution signal which indicates const
    ir_h = env.ir_h
    if isinstance(py_value, np.ndarray):
      dense_attr = ir_h.context.dense_elements_attr(py_value)
      tensor_type = dense_attr.type
      tensor_value = ir_h.constant_op(tensor_type, dense_attr).result
      return ir_h.numpy_create_array_from_tensor_op(tensor_value).result
    return NotImplemented


def CreateNumpyValueCoder() -> ValueCoder:
  return ValueCoderChain((NdArrayValueCoder(),))
