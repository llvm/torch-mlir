#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from npcomp.native.mlir import ir

__all__ = [
  "load_builtin_module",
  "Types",
]


class Ops(ir.Ops):
  r"""Dialect ops.
  
    >>> c = ir.MLIRContext()
    >>> t = Types(c)
    >>> m = c.new_module()
    >>> tensor_type = t.tensor(t.f32)
    >>> ops = Ops(c)
    >>> ops.builder.insert_block_start(m.first_block)
    >>> f = ops.func_op("foobar", t.function(
    ...   [tensor_type, tensor_type], [tensor_type]), 
    ...   create_entry_block=True)
    >>> uf = ops.numpy_ufunc_call_op("numpy.add", tensor_type,
    ...   *f.first_block.args)
    >>> _ = ops.return_op(uf.results)
    >>> print(m.to_asm())
    <BLANKLINE>
    <BLANKLINE>
    module {
      func @foobar(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
        %0 = numpy.ufunc_call @numpy.add(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
        return %0 : tensor<*xf32>
      }
    }
  """
  def numpy_ufunc_call_op(self, callee_symbol, result_type, *args):
    """Creates a numpy.ufunc_call op."""
    c = self.context
    attrs = c.dictionary_attr({
      "ufunc_ref": c.flat_symbol_ref_attr(callee_symbol)
    })
    return self.op("numpy.ufunc_call", [result_type], args, attrs)

  def numpy_narrow(self, result_type, operand):
    """Creates a numpy.narrow op."""
    return self.op("numpy.narrow", [result_type], [operand])


class Types(ir.Types):
  """Container/factory for dialect types.

    >>> t = Types(ir.MLIRContext())
    >>> t.numpy_any_dtype
    !numpy.any_dtype
    >>> t.tensor(t.numpy_any_dtype, [1, 2, 3])
    tensor<1x2x3x!numpy.any_dtype>
    >>> t.tensor(t.numpy_any_dtype)
    tensor<*x!numpy.any_dtype>
    >>> t.tensor(t.numpy_any_dtype, [-1, 2])
    tensor<?x2x!numpy.any_dtype>
    >>> t.tensor(t.f32)
    tensor<*xf32>
    >>> t.function([t.i32], [t.f32])
    (i32) -> f32

  """
  def __init__(self, context):
    super().__init__(context)
    self.numpy_any_dtype = context.parse_type("!numpy.any_dtype")


def load_builtin_module(context=None):
  """Loads a module populated with numpy built-ins.

  This is not a long-term solution but overcomes some bootstrapping
  issues.

    >>> m = load_builtin_module()
    >>> op = m.region(0).blocks.front.operations.front
    >>> op.is_registered
    True
    >>> op.name
    'numpy.generic_ufunc'

  Args:
    context: The MLIRContext to use (None to create a new one).
  Returns:
    A ModuleOp.
  """
  if context is None: context = ir.MLIRContext()
  return context.parse_asm(_BUILTIN_MODULE_ASM)


_BUILTIN_MODULE_ASM = r"""
  numpy.generic_ufunc @numpy.add (
    overload(%arg0: i32, %arg1: i32) -> i32 {
      %0 = addi %arg0, %arg1 : i32
      numpy.ufunc_return %0 : i32
    },
    overload(%arg0: f32, %arg1: f32) -> f32 {
      %0 = addf %arg0, %arg1 : f32
      numpy.ufunc_return %0 : f32
    }
  )
  numpy.generic_ufunc @numpy.multiple (
    overload(%arg0: i32, %arg1: i32) -> i32 {
      %0 = muli %arg0, %arg1 : i32
      numpy.ufunc_return %0 : i32
    },
    overload(%arg0: f32, %arg1: f32) -> f32 {
      %0 = mulf %arg0, %arg1 : f32
      numpy.ufunc_return %0 : f32
    }
  )
"""

if __name__ == "__main__":
  import doctest
  doctest.testmod()
