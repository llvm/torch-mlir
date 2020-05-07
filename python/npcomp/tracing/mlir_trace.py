#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
from typing import Iterable
import numpy as np

from _npcomp.mlir import ir

from npcomp.dialect import Numpy
from npcomp.exporter import *
from npcomp.types import *
from npcomp.tracing.context import *
from npcomp.tracing.emitters import *


class ModuleBuilder:
  """Builds an MLIR module by tracing functions."""

  def __init__(self, mlir_context=None, emitter_registry=None):
    self.context = mlir_context if mlir_context else ir.MLIRContext()
    # TODO: Instead of bootstrapping a large module, populate imports
    # dynamically.
    self.module = Numpy.load_builtin_module(self.context)
    self.helper = Numpy.DialectHelper(self.context)
    self.emitters = (emitter_registry
                     if emitter_registry else EmitterRegistry.create_default())

  def trace(self, *export_py_funcs: ExportPyFunction):
    """Traces exported py functions."""
    for export_py_func in export_py_funcs:
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
      "_helper",
      "_mlir_m",
      "_mlir_c",
      "_python_args",
      "_result_array_params",
      "_traced_arrays",
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
    self._helper = module_builder.helper

    # Extract ArrayParams for all args and results.
    self._args_array_params = [
        ArrayParams.from_constraints(arg.constraints)
        for arg in self.epf.sig.args
    ]
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
    h = self._helper
    py_results = (self.epf.pyfunc(*self._python_args),)
    if len(py_results) != len(self._f_types):
      raise TracingError("Traced function returned != %d results: %r" % (
          len(self._f_types),
          py_results,
      ))

    # Narrow all results to the declared return types.
    return_operands = []
    for py_result, mlir_result_type in zip(py_results, self._f_types):
      mlir_result = self.get_traced_array_value(py_result)
      if mlir_result is None:
        raise TracingError("Unregistered traced array: %r", (py_result,))
      # narrow to declared result type.
      return_operands.extend(
          h.numpy_narrow_op(mlir_result_type, mlir_result).results)
    h.return_op(return_operands)

  def set_traced_array(self, traced_array, value):
    """Sets the current SSA value for a traced_array."""
    assert isinstance(traced_array, TracedArray)
    self._traced_arrays[traced_array] = value

  def get_traced_array_value(self, traced_array):
    return self._traced_arrays.get(traced_array)

  def _validate(self):
    if not all(
        arg.type_class == TypeClass.NdArray for arg in self.epf.sig.args):
      raise NotImplementedError("Non NdArray args: %r" % (self.epf.sig.args,))
    if not self.epf.sig.result.type_class == TypeClass.NdArray:
      raise NotImplementedError("Non NdArray result: %r" %
                                (self.epf.sig.result,))

  def _create_mlir_function(self):
    mlir_c = self._mlir_c
    mlir_m = self._mlir_m
    h = self._helper
    epf = self.epf
    f_args = [
        mlir_c.parse_type(ap.mlir_tensor_type_asm)
        for ap in self._args_array_params
    ]
    f_types = [
        mlir_c.parse_type(self._result_array_params.mlir_tensor_type_asm)
    ]
    h.builder.insert_before_terminator(mlir_m.first_block)
    f_type = h.function_type(f_args, f_types)
    f = h.func_op(epf.__name__, f_type, create_entry_block=True)
    return f, f_types

  def _create_trace_roots(self):
    entry_block = self._f.first_block
    for index, ap in enumerate(self._args_array_params):
      if ap is not None:
        ta = TracedArray(self)
        self.set_traced_array(ta, entry_block.args[index])
        self._python_args[index] = ta

  def _resolve_input_ssa_values(self, trace_values: Iterable[TraceValue]):
    """Resolves input python values to SSA values."""
    ssa_values = []
    for tv in trace_values:
      assert tv.type == TraceValueType.NDARRAY, (
          "Unsupported TraceValueType: %r" % tv.type)
      ssa_value = self.get_traced_array_value(tv.value)
      if ssa_value is None:
        raise TracingError(
            "Required a traced python NDARRAY but not found: %r" % (tv,))
      ssa_values.append(ssa_value)
    return ssa_values

  def _resolve_result_py_values(self,
                                trace_value_types: Iterable[TraceValueType],
                                ssa_values):
    """Resolves result SSA values to runtime python values."""
    assert len(trace_value_types) == len(ssa_values), (
        "Mismatched emitter declared result types and results")
    py_values = []
    for trace_value_type, ssa_value in zip(trace_value_types, ssa_values):
      assert trace_value_type == TraceValueType.NDARRAY, (
          "Unsupported TraceValueType: %r" % trace_value_type)
      py_value = TracedArray(self)
      self.set_traced_array(py_value, ssa_value)
      py_values.append(py_value)
    return py_values

  def _emit_invocation(self, emitter: FuncEmitter, invocation: TraceInvocation):
    tv_map = emitter.map_invocation(invocation)
    input_ssa_values = self._resolve_input_ssa_values(tv_map.input_trace_values)
    request = EmissionRequest(input_ssa_values,
                              dialect_helper=self._helper,
                              extra=tv_map.extra)
    result_ssa_values = emitter.emit(request)
    py_values = self._resolve_result_py_values(tv_map.result_trace_value_types,
                                               result_ssa_values)
    return emitter.map_results(py_values, tv_map.extra)

  def _handle_ufunc(self, ufunc, method, inputs, kwargs):
    emitter = self.module_builder.emitters.lookup_ufunc(ufunc, method)
    if not emitter:
      return NotImplemented
    invocation = TraceInvocation(inputs, kwargs, Protocol.UFUNC, method)
    return self._emit_invocation(emitter, invocation)

  def _handle_array_func(self, func, types, inputs, kwargs):
    emitter = self.module_builder.emitters.lookup_array_func(func)
    if not emitter:
      return NotImplemented
    invocation = TraceInvocation(inputs, kwargs, Protocol.ARRAY_FUNC)
    return self._emit_invocation(emitter, invocation)


if __name__ == "__main__":
  import doctest
  doctest.testmod()
