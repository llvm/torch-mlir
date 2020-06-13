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
    >>> m = c.new_module()
    >>> tensor_type = h.tensor_type(h.f32_type)
    >>> h.builder.insert_block_start(m.first_block)
    >>> f = h.func_op("foobar", h.function_type(
    ...   [tensor_type, tensor_type], [tensor_type]), 
    ...   create_entry_block=True)
    >>> uf = h.numpy_ufunc_call_op("numpy.add", tensor_type,
    ...   *f.first_block.args)
    >>> _ = h.return_op(uf.results)
    >>> print(m.to_asm())
    <BLANKLINE>
    <BLANKLINE>
    module {
      func @foobar(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
        %0 = numpy.ufunc_call @numpy.add(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
        return %0 : tensor<*xf32>
      }
    }

  DenseElementsAttrs:
    >>> c.dense_elements_attr(np.asarray([1, 2, 3, 4]))
    dense<[1, 2, 3, 4]> : tensor<4xsi64>
    >>> c.dense_elements_attr(np.asarray([[1, 2], [3, 4]]))
    dense<[[1, 2], [3, 4]]> : tensor<2x2xsi64>
    >>> c.dense_elements_attr(np.asarray([[1., 2.], [3., 4.]]))
    dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
    >>> c.dense_elements_attr(np.asarray([[1., 2.], [3., 4.]], dtype=np.float32))
    dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>

  Types:
    >>> c = ir.MLIRContext()
    >>> t = DialectHelper(c, ir.OpBuilder(c))
    >>> t.numpy_any_dtype
    !numpy.any_dtype
    >>> t.tensor_type(t.numpy_any_dtype, [1, 2, 3])
    tensor<1x2x3x!numpy.any_dtype>
    >>> t.tensor_type(t.numpy_any_dtype)
    tensor<*x!numpy.any_dtype>
    >>> t.tensor_type(t.numpy_any_dtype, [-1, 2])
    tensor<?x2x!numpy.any_dtype>
    >>> t.tensor_type(t.f32_type)
    tensor<*xf32>
    >>> t.function_type([t.i32_type], [t.f32_type])
    (i32) -> f32
    >>> t.unknown_array_type
    tensor<*x!numpy.any_dtype>

  """

  @property
  def numpy_any_dtype(self):
    return self.context.parse_type("!numpy.any_dtype")

  @property
  def unknown_array_type(self):
    return self.tensor_type(self.numpy_any_dtype)

  def numpy_ufunc_call_op(self, callee_symbol, result_type, *args):
    """Creates a numpy.ufunc_call op."""
    c = self.context
    attrs = c.dictionary_attr(
        {"ufunc_ref": c.flat_symbol_ref_attr(callee_symbol)})
    return self.op("numpy.ufunc_call", [result_type], args, attrs)

  def numpy_narrow_op(self, result_type, operand):
    """Creates a numpy.narrow op."""
    return self.op("numpy.narrow", [result_type], [operand])

  def numpy_get_slice_op(self, result_type, array, *slice_elements):
    return self.op("numpy.get_slice", [result_type],
                   [array] + list(slice_elements))


def load_builtin_module(context=None):
  """Loads a module populated with numpy built-ins.

  This is not a long-term solution but overcomes some bootstrapping
  issues.

    >>> m = load_builtin_module()
    >>> op = m.region(0).blocks.front.operations.front
    >>> op.is_registered
    True
    >>> op.name
    'numpy.builtin_ufunc'

  Args:
    context: The MLIRContext to use (None to create a new one).
  Returns:
    A ModuleOp.
  """
  if context is None:
    context = ir.MLIRContext()
  return context.parse_asm(_BUILTIN_MODULE_ASM)


_BUILTIN_MODULE_ASM = r"""
  numpy.builtin_ufunc @numpy.add
  numpy.builtin_ufunc @numpy.multiply
"""

if __name__ == "__main__":
  import doctest
  doctest.testmod()
