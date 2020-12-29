#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
from typing import Iterable, Optional
import numpy as np

from mlir import ir as _ir
from mlir.dialects import std as std_ops

from npcomp import _cext
from npcomp.dialects import basicpy as basicpy_ops
from npcomp.dialects import numpy as numpy_ops

from ..exporter import *
from ..types import *
from ..compiler.utils.mlir_utils import *

from .context import *
from .emitters import *


class ModuleBuilder:
  """Builds an MLIR module by tracing functions."""

  __slots__ = [
      "emitters",
      "ic",
  ]

  def __init__(self,
               mlir_context: Optional[_ir.Context] = None,
               emitter_registry=None):
    ic = self.ic = ImportContext(mlir_context)
    ic.module = _ir.Module.create(loc=ic.loc)
    self.emitters = (emitter_registry
                     if emitter_registry else EmitterRegistry.create_default())

  @property
  def module(self):
    return self.ic.module

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
      "_ic",
      "_python_args",
      "_result_array_params",
      "_traced_arrays",
      "_external_arrays",
  ]

  def __init__(self, module_builder: ModuleBuilder, epf: ExportPyFunction):
    super().__init__(desc="[trace of %s]" % epf.__name__)
    self.module_builder = module_builder
    self.epf = epf
    self._traced_arrays = {}  # Mapping of TracedArray to current consumer value
    self._external_arrays = {}  # Mapping of id to (ndarray, ir.Value)
    self._validate()

    # Alias some parent members for convenience.
    self._ic = module_builder.ic
    with self._ic.context:
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

  @property
  def entry_block(self) -> _ir.Block:
    return self._f.regions[0].blocks[0]

  def trace(self):
    # Invoke the python function with placeholders.
    # TODO: More sophisticated signature merging
    # TODO: Multiple results
    # TODO: Error reporting
    ic = self._ic
    ic.insert_end_of_block(self.entry_block)
    with ic.context:
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
      # narrow to declared result type.
      return_operands.extend(
          numpy_ops.NarrowOp(mlir_result_type,
                             mlir_result,
                             loc=ic.loc,
                             ip=ic.ip).results)
    std_ops.ReturnOp(return_operands, loc=ic.loc, ip=ic.ip)
    ic.pop_ip()

  def set_traced_array(self, traced_array, value):
    """Sets the current SSA value for a traced_array."""
    assert isinstance(traced_array, TracedArray)
    self._traced_arrays[traced_array] = value

  def get_traced_array_value(self, traced_array):
    if not isinstance(traced_array, TracedArray):
      # Generic import of external value. For now, we just treat these as
      # local consts.
      return self._get_external_array_value(traced_array)

    traced_value = self._traced_arrays.get(traced_array)
    if traced_value is None:
      raise TracingError("Unregistered traced array: %r", (traced_array,))
    return traced_value

  def _get_external_array_value(self, external_array):
    ic = self._ic
    if not isinstance(external_array, np.ndarray):
      raise TracingError("Expected ndarray but got: %r" % (external_array,))
    found_it = self._external_arrays.get(id(external_array))
    if found_it:
      return found_it[1]
    # Import it.
    dense_attr = _ir.DenseElementsAttr.get(external_array, context=ic.context)
    const_value = std_ops.ConstantOp(dense_attr.type,
                                     dense_attr,
                                     loc=ic.loc,
                                     ip=ic.ip).result
    self._external_arrays[id(external_array)] = (external_array, const_value)
    return const_value

  def _validate(self):
    if not all(
        arg.type_class == TypeClass.NdArray for arg in self.epf.sig.args):
      raise NotImplementedError("Non NdArray args: %r" % (self.epf.sig.args,))
    if not self.epf.sig.result.type_class == TypeClass.NdArray:
      raise NotImplementedError("Non NdArray result: %r" %
                                (self.epf.sig.result,))

  def _create_mlir_function(self):
    ic = self._ic
    epf = self.epf
    f_args = [
        _ir.Type.parse(ap.mlir_tensor_type_asm)
        for ap in self._args_array_params
    ]
    f_types = [_ir.Type.parse(self._result_array_params.mlir_tensor_type_asm)]
    ic.insert_before_terminator(ic.module.body)
    f_type = _ir.FunctionType.get(f_args, f_types)
    f, _ = ic.FuncOp(epf.__name__, f_type, create_entry_block=True)
    return f, f_types

  def _create_trace_roots(self):
    entry_block = self.entry_block
    for index, ap in enumerate(self._args_array_params):
      if ap is not None:
        ta = TracedArray(self)
        self.set_traced_array(ta, entry_block.arguments[index])
        self._python_args[index] = ta

  def _resolve_input_ssa_values(self, trace_values: Iterable[TraceValue]):
    """Resolves input python values to SSA values."""
    ssa_values = []
    for tv in trace_values:
      assert tv.type == TraceValueType.NDARRAY, (
          "Unsupported TraceValueType: %r" % tv.type)
      ssa_value = self.get_traced_array_value(tv.value)
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
    request = EmissionRequest(input_ssa_values, ic=self._ic, extra=tv_map.extra)
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

  def _emit_slice_value(self, slice_element):
    ic = self._ic
    if slice_element == None:
      return basicpy_ops.SingletonOp(ic.none_type, loc=ic.loc, ip=ic.ip).result
    elif slice_element == Ellipsis:
      return basicpy_ops.SingletonOp(ic.ellipsis_type, loc=ic.loc,
                                     ip=ic.ip).result
    elif isinstance(slice_element, int):
      return std_ops.ConstantOp(ic.index_type,
                                _ir.IntegerAttr.get(ic.index_type,
                                                    slice_element),
                                loc=ic.loc,
                                ip=ic.ip).result
    elif isinstance(slice_element, slice):
      return self._emit_slice_object(slice_element)
    else:
      # Assume array convertible.
      raise NotImplementedError(
          "TODO: Slicing with generic arrays not yet implemented")

  def _emit_slice_object(self, slice_object: slice):
    ic = self._ic

    def emit_index(index):
      if index is None:
        return basicpy_ops.SingletonOp(ic.none_type, loc=ic.loc,
                                       ip=ic.ip).result
      else:
        return std_ops.ConstantOp(ic.index_type,
                                  _ir.IntegerAttr.get(ic.index_type,
                                                      int(index)),
                                  loc=ic.loc,
                                  ip=ic.ip).result

    start = emit_index(slice_object.start)
    stop = emit_index(slice_object.stop)
    step = emit_index(slice_object.step)
    result_type = _cext.slot_object_type(ic.context, "slice",
                                         [start.type, stop.type, step.type])
    return basicpy_ops.SlotObjectMakeOp(result_type, [start, stop, step],
                                        loc=ic.loc,
                                        ip=ic.ip).result

  def _handle_array_getitem(self, array, key):
    ic = self._ic
    array_value = self.get_traced_array_value(array)
    # Array slicing is always based on a tuple.
    slice_tuple = key if isinstance(key, tuple) else (key,)
    # Resolve and emit each slice element.
    slice_values = [self._emit_slice_value(elt) for elt in slice_tuple]
    result_value = numpy_ops.GetSliceOp(ic.unknown_array_type,
                                        array_value,
                                        slice_values,
                                        loc=ic.loc,
                                        ip=ic.ip).result
    result_array = TracedArray(self)
    self.set_traced_array(result_array, result_value)
    return result_array


if __name__ == "__main__":
  import doctest
  doctest.testmod()
