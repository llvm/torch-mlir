#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
from npcomp.dialect import Basicpy
from _npcomp.mlir import ir

__all__ = [
    "load_builtin_module",
    "DialectHelper",
]


class DialectHelper(Basicpy.DialectHelper):
  r"""Dialect helper.

    >>> c = ir.MLIRContext()
    >>> h = DialectHelper(c, ir.OpBuilder(c))

  DenseElementsAttrs:
    >>> c.dense_elements_attr(np.asarray([1, 2, 3, 4], dtype=np.int32))
    dense<[1, 2, 3, 4]> : tensor<4xsi32>
    >>> c.dense_elements_attr(np.asarray([[1, 2], [3, 4]], dtype=np.int32))
    dense<[[1, 2], [3, 4]]> : tensor<2x2xsi32>
    >>> c.dense_elements_attr(np.asarray([[1., 2.], [3., 4.]]))
    dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
    >>> c.dense_elements_attr(np.asarray([[1., 2.], [3., 4.]], dtype=np.float32))
    dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>

  Types:
    >>> c = ir.MLIRContext()
    >>> t = DialectHelper(c, ir.OpBuilder(c))
    >>> t.numpy_any_dtype
    !basicpy.UnknownType
    >>> t.tensor_type(t.numpy_any_dtype, [1, 2, 3])
    tensor<1x2x3x!basicpy.UnknownType>
    >>> t.tensor_type(t.numpy_any_dtype)
    tensor<*x!basicpy.UnknownType>
    >>> t.tensor_type(t.numpy_any_dtype, [-1, 2])
    tensor<?x2x!basicpy.UnknownType>
    >>> t.tensor_type(t.f32_type)
    tensor<*xf32>
    >>> t.function_type([t.i32_type], [t.f32_type])
    (i32) -> f32
    >>> t.numpy_unknown_tensor_type
    tensor<*x!basicpy.UnknownType>

  """

  @property
  def numpy_any_dtype(self):
    return self.basicpy_UnknownType

  @property
  def numpy_unknown_tensor_type(self):
    return self.tensor_type(self.basicpy_UnknownType)

  @property
  def unknown_array_type(self):
    return self.numpy_NdArrayType(self.basicpy_UnknownType)

  def numpy_builtin_ufunc_call_op(self, *args, qualified_name, result_type):
    """Creates a numpy.builtin_ufunc_call op."""
    c = self.context
    attrs = c.dictionary_attr({"qualified_name": c.string_attr(qualified_name)})
    return self.op("numpy.builtin_ufunc_call", [result_type], args, attrs)

  def numpy_narrow_op(self, result_type, operand):
    """Creates a numpy.narrow op."""
    return self.op("numpy.narrow", [result_type], [operand])

  def numpy_get_slice_op(self, result_type, array, *slice_elements):
    return self.op("numpy.get_slice", [result_type],
                   [array] + list(slice_elements))


if __name__ == "__main__":
  import doctest
  doctest.testmod()
