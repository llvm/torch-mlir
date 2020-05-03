#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re

import numpy as np

from ..dialect import Numpy
from ..native.mlir import ir

from .context import *
from ..exporter import *
from ..types import *


class ModuleBuilder:
  """Builds an MLIR module by tracing functions."""
  def __init__(self, mlir_context=None):
    self.context = context if mlir_context else ir.MLIRContext()
    # TODO: Instead of bootstrapping a large module, populate imports
    # dynamically.
    self.module = Numpy.load_builtin_module(self.context)
    self.ops = Numpy.Ops(self.context)
    self.types = Numpy.Types(self.context)

  def trace(self, export_py_func: ExportPyFunction):
    """Traces and exported python function."""
    assert isinstance(export_py_func, ExportPyFunction), (
      "Expected an exported python function (from the Exporter class)")
    tracer = FunctionTracer(self, export_py_func)
    with tracer:
      tracer.trace()


class FunctionTracer(TraceContext):
  """A trace of a single function."""
  __slots__ = [
    "module_builder",
    "epf",
    "_args_array_params",
    "_f",
    "_f_types",
    "_mlir_m",
    "_mlir_c",
    "_python_args",
    "_ops",
    "_result_array_params",
    "_traced_arrays",
    "_types",
  ]
  def __init__(self, module_builder: ModuleBuilder, epf: ExportPyFunction):
    super().__init__(desc="[trace of %s]" % epf.__name__)
    self.module_builder = module_builder
    self.epf = epf
    self._traced_arrays = {}  # Mapping of TracedArray to current consumer value
    self._validate()

    # Alias some parent members for convenience.
    self._mlir_m = module_builder.module
    self._mlir_c = module_builder.context
    self._ops = module_builder.ops
    self._types = module_builder.types

    # Extract ArrayParams for all args and results.
    self._args_array_params = [
      ArrayParams.from_constraints(arg.constraints) 
      for arg in self.epf.sig.args]
    self._python_args = [None] * len(self._args_array_params)
    self._result_array_params = ArrayParams.from_constraints(
      self.epf.sig.result.constraints)    

    # Create the MLIR function.
    self._f, self._f_types = self._create_mlir_function()
    self._create_trace_roots()

  def trace(self):
    # Invoke the python function with placeholders.
    # TODO: More sophisticated signature merging
    # TODO: Multiple results
    # TODO: Error reporting
    ops = self._ops
    py_results = (self.epf.pyfunc(*self._python_args),)
    if len(py_results) != len(self._f_types):
      raise TracingError(
        "Traced function returned != %d results: %r" % (
          len(self._f_types), py_results,))
        
    # Narrow all results to the declared return types.
    return_operands = []
    for py_result, mlir_result_type in zip(py_results, self._f_types):
      mlir_result = self.get_traced_array_value(py_result)
      if mlir_result is None:
        raise TracingError("Unregistered traced array: %r", (py_result,))
      # narrow to declared result type.
      return_operands.extend(
        ops.numpy_narrow(mlir_result_type, mlir_result).results)
    ops.return_op(return_operands)

  def set_traced_array(self, traced_array, value):
    """Sets the current SSA value for a traced_array."""
    assert isinstance(traced_array, TracedArray)
    self._traced_arrays[traced_array] = value

  def get_traced_array_value(self, traced_array):
    return self._traced_arrays.get(traced_array)

  def _validate(self):
    if not all(arg.type_class == TypeClass.NdArray 
               for arg in self.epf.sig.args):
      raise NotImplementedError("Non NdArray args: %r" % (self.epf.sig.args,))
    if not self.epf.sig.result.type_class == TypeClass.NdArray:
      raise NotImplementedError("Non NdArray result: %r" % (
        self.epf.sig.result,))

  def _create_mlir_function(self):
    mlir_c = self._mlir_c
    mlir_m = self._mlir_m
    ops = self._ops
    types = self._types
    epf = self.epf
    f_args = [mlir_c.parse_type(ap.mlir_tensor_type_asm)
              for ap in self._args_array_params]
    f_types = [mlir_c.parse_type(
      self._result_array_params.mlir_tensor_type_asm)]
    ops.builder.insert_before_terminator(mlir_m.first_block)
    f_type = types.function(f_args, f_types)
    f = ops.func_op(epf.__name__, f_type, create_entry_block=True)
    return f, f_types

  def _create_trace_roots(self):
    entry_block = self._f.first_block
    for index, ap in enumerate(self._args_array_params):
      if ap is not None:
        ta = TracedArray(self)
        self.set_traced_array(ta, entry_block.args[index])
        self._python_args[index] = ta

  def _handle_ufunc(self, ufunc, method, inputs, kwargs):
    if method == "__call__":
      if kwargs:
        raise TracingError("Generic ufunc with kwargs not supported %r" % (
          ufunc,))
      
      # Map inputs to TracedArrays.
      # TODO: Process captures, promotions, etc.
      op_inputs = []
      for py_input in inputs:
        if not isinstance(py_input, TracedArray):
          raise TracingError("Unsupported ufunc input: %r", (py_input,))
        op_input = self.get_traced_array_value(py_input)
        if op_input is None:
          raise TracingError("Unregistered traced array: %r", (py_input,))
        op_inputs.append(op_input)
      
      # Emit op.
      types = self._types
      mlir_m = self._mlir_m
      callee_symbol = _UFUNC_SYMBOL_MAP.get(ufunc)
      if not callee_symbol:
        raise TracingError("Unsupported ufunc: %r" % ufunc)
      op_result_type = types.tensor(types.numpy_any_dtype)
      call_op = self._ops.numpy_ufunc_call_op(
        callee_symbol, op_result_type, *op_inputs)
      op_result = call_op.results[0]
      
      # Wrap returns.
      return_array = TracedArray(self)
      self.set_traced_array(return_array, op_result)
      return return_array

    # Unsupported method.
    raise TracingError("Unsupported ufunc method %r:%r" % (ufunc, method,))


# TODO: There should be an open registry of ufuncs. But for now, just map
# introspect the numpy package and record them.
def _build_ufunc_symbol_map():
  d = {}
  for member in dir(np):
    ufunc = getattr(np, member)
    if isinstance(ufunc, np.ufunc):
      d[ufunc] = "numpy." + member
  return d

_UFUNC_SYMBOL_MAP = _build_ufunc_symbol_map()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
