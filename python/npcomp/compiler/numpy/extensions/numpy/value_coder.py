#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Value coders for Numpy types."""

import numpy as np
from typing import Union

from mlir import ir as _ir
from mlir.dialects import std as std_ops

from npcomp import _cext
from npcomp.dialects import numpy as numpy_ops

from ....utils import logging
from ...interfaces import *

__all__ = [
    "CreateNumpyValueCoder",
]

_NotImplementedType = type(NotImplemented)


class NdArrayValueCoder(ValueCoder):
  """Value coder for numpy types."""
  __slots__ = []

  def code_py_value_as_const(self, env: Environment,
                             py_value) -> Union[_NotImplementedType, _ir.Value]:
    # TODO: Query for ndarray compat (for duck typed and such)
    # TODO: Have a higher level name resolution signal which indicates const
    ic = env.ic
    if isinstance(py_value, np.ndarray):
      dense_attr = _ir.DenseElementsAttr.get(py_value, context=ic.context)
      tensor_type = dense_attr.type
      tensor_value = std_ops.ConstantOp(tensor_type,
                                        dense_attr,
                                        loc=ic.loc,
                                        ip=ic.ip).result
      ndarray_type = _cext.shaped_to_ndarray_type(tensor_type)
      return numpy_ops.CreateArrayFromTensorOp(ndarray_type,
                                               tensor_value,
                                               loc=ic.loc,
                                               ip=ic.ip).result
    return NotImplemented


def CreateNumpyValueCoder() -> ValueCoder:
  return ValueCoderChain((NdArrayValueCoder(),))
