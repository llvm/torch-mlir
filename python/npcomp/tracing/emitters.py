#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np

from collections import namedtuple
from enum import Enum


class Protocol(Enum):
  UFUNC = 1
  ARRAY_FUNC = 2


class TraceValueType(Enum):
  NDARRAY = 1


class TraceValue(
    namedtuple("TraceValue", ["value", "type"],
               defaults=(TraceValueType.NDARRAY,))):
  __slots__ = ()
  """A Python value and the trace type that it should correspond to."""


class TraceInvocation(
    namedtuple("TraceInvocation", ["inputs", "kwargs", "protocol", "method"],
               defaults=(Protocol.ARRAY_FUNC, "__call__"))):
  """An invocation of a single functions.

  This abstracts over both ufuncs and array_funcs, differentiating by the
  protocol and method.
  """
  __slots__ = ()


class EmissionRequest(
    namedtuple("EmissionRequest",
               ["input_ssa_values", "dialect_helper", "extra"],
               defaults=(None,))):
  """Represents the result of processing inputs from an invocation.

  The `input_ssa_values` are mlir.ir.Value instances corresponding to
  input_trace_values in TraceValueMap.

  The `extra` value is only relevant to the producer and can be used as a
  blackbox mechanism to transfer un-tracked state from an invocation to
  emission.

  The `dialect_helper` fields correspond to mlir.ir.DialectHelper.
  """
  __slots__ = ()


class TraceValueMap(
    namedtuple("TraceValueMap",
               ["input_trace_values", "result_trace_value_types", "extra"],
               defaults=(None,))):
  """The result of mapping an invocation to corresponding op structure.

  This type associates:
    - Python (object, TraceValueType) representing invocation inputs that
      correspond to SSA values in the IR.
    - TraceValueTypes that are the expected logical result types from the
      invocation.
    - 'extra' object that is passed to followon Emitter methods.
  """
  __slots__ = ()


class FuncEmitter:
  """An emitter for an op-like function invocation."""

  def map_invocation(self, trace_invocation: TraceInvocation) -> TraceValueMap:
    """Maps from an invocation to EmissionRequest.

    This hook is also responsible for validating the invocation and should
    raise appropriate user-visible exceptions (i.e. when invoked with incorrect
    arguments).

    This hook is used to prepare for emission in a define-by-run scenario.
    Static emission from an AST needs to be prepared via another mechanism.

    Args:
      trace_invocation: An Invocation instance to map.
    Returns:
      A TraceValueMap describing the structure of the invocation as mapped
      to/from IR.
    """
    raise NotImplementedError()

  def map_results(self, py_results, extra):
    """Maps a list of python results to actual function return values.

    Args:
      py_results: List of python results corresponding to the emitted op
        results.
      extra: The extra object returned by map_invocation.
    Returns:
      Actual function result. Typically this requires special handling to
      unpack the result of functions that return 1 item.
    """
    raise NotImplementedError()

  def emit(self, request: EmissionRequest):
    """Emits IR using the provided ops and types factories.

    Args:
      emission_inputs: An EmissionRequest produced by tracing each TraceValue
        from a previous call to map_invocation and the corresponding extra
        value.
    Returns:
      An iterable of mlir.ir.Value instances representing the outputs of the
      operation. The `builder` on `ops` must be positioned to consume these
      values.
    """
    raise NotImplementedError()


class GenericCallUfuncEmitter(FuncEmitter):
  """A FuncEmitter for generic ufuncs requiring no special behavior.

  Representation:
    >>> emitter = GenericCallUfuncEmitter("numpy.add")
    >>> emitter
    <ufunc emitter 'numpy.add'>
    >>> inv = TraceInvocation([1, 2], {}, protocol=Protocol.UFUNC)
    >>> inputs = emitter.map_invocation(inv)
    >>> inputs
    TraceValueMap(input_trace_values=[TraceValue(value=1, type=<TraceValueType.NDARRAY: 1>), TraceValue(value=2, type=<TraceValueType.NDARRAY: 1>)], result_trace_value_types=[<TraceValueType.NDARRAY: 1>], extra=None)

  Error on unsupported kwargs:
    >>> inv = TraceInvocation([1, 2], {"foobar": 1}, protocol=Protocol.UFUNC)
    >>> emitter.map_invocation(inv)
    Traceback (most recent call last):
    ...
    ValueError: Unexpected keyword args for ufunc numpy.add: foobar

  """
  __slots__ = ("_ufunc_name")

  def __init__(self, ufunc_name: str):
    self._ufunc_name = ufunc_name

  def __repr__(self):
    return "<ufunc emitter '%s'>" % self._ufunc_name

  def map_invocation(self,
                     trace_invocation: TraceInvocation) -> EmissionRequest:
    assert trace_invocation.protocol == Protocol.UFUNC
    assert trace_invocation.method == "__call__"
    if trace_invocation.kwargs:
      raise ValueError(
          "Unexpected keyword args for ufunc %s: %s" %
          (self._ufunc_name, ", ".join(trace_invocation.kwargs.keys())))
    # Without above special cases, any positional args map to emission
    # inputs.
    return TraceValueMap([
        TraceValue(i, TraceValueType.NDARRAY) for i in trace_invocation.inputs
    ], [TraceValueType.NDARRAY],
                         extra=None)

  def map_results(self, py_results, extra):
    # Ufuncs always return one result, so just unpack it.
    return py_results[0]

  def emit(self, request: EmissionRequest):
    h = request.dialect_helper
    op_result_type = h.tensor_type(h.numpy_any_dtype)
    call_op = h.numpy_builtin_ufunc_call_op(*request.input_ssa_values,
                                            qualified_name=self._ufunc_name,
                                            result_type=op_result_type)
    return call_op.results


class GenericArrayFuncEmitter(FuncEmitter):
  """Emitter for array funcs that don't do anything 'special'."""
  __slots__ = ("_op_name", "_nresults")

  def __init__(self, op_name: str, nresults: int = 1):
    self._op_name = op_name
    self._nresults = nresults

  def __repr__(self):
    return "<array_func emitter '%s'>" % self._op_name

  def map_invocation(self,
                     trace_invocation: TraceInvocation) -> EmissionRequest:
    assert trace_invocation.protocol == Protocol.ARRAY_FUNC
    if trace_invocation.method != "__call__":
      raise NotImplementedError("Only __call__ is supported for %s (got '%s')" %
                                (
                                    self._op_name,
                                    trace_invocation.method,
                                ))
    if trace_invocation.kwargs:
      raise ValueError(
          "Unexpected keyword args for %s: %s" %
          (self._op_name, ", ".join(trace_invocation.kwargs.keys())))
    # Without above special cases, any positional args map to emission
    # inputs.
    return TraceValueMap([
        TraceValue(i, TraceValueType.NDARRAY) for i in trace_invocation.inputs
    ], [TraceValueType.NDARRAY] * self._nresults,
                         extra=None)

  def map_results(self, py_results, extra):
    if self._nresults == 1:
      return py_results[0]
    else:
      return tuple(py_results)

  def emit(self, request: EmissionRequest):
    h = request.dialect_helper
    op_result_types = [h.tensor_type(h.numpy_any_dtype)] * self._nresults
    op = h.op(self._op_name, op_result_types, request.input_ssa_values)
    return op.results


class EmitterRegistry:
  """Registry of known Emitter instances mapped to source function.

    >>> r = EmitterRegistry.create_default()
    >>> r.lookup_ufunc(np.add, "__call__")
    <ufunc emitter 'numpy.add'>
    >>> r.lookup_array_func(np.dot)
    <array_func emitter 'numpy.dot'>
  """

  def __init__(self):
    self._ufunc_map = {}  # Dictionary of (f, method) -> Emitter
    self._arrayfunc_map = {}  # Dictionary of f -> Emitter

  @classmethod
  def create_default(cls):
    registry = cls()
    registry.register_defaults()
    return registry

  def register_ufunc(self, ufunc, method, emitter):
    # Last registration wins.
    self._ufunc_map[(ufunc, method)] = emitter

  def register_array_func(self, f, emitter):
    # Last registration wins.
    self._arrayfunc_map[f] = emitter

  def lookup_ufunc(self, ufunc, method):
    return self._ufunc_map.get((ufunc, method))

  def lookup_array_func(self, f):
    return self._arrayfunc_map.get(f)

  def register_defaults(self):
    # Find all ufuncs in the numpy module and register by name.
    for member in sorted(dir(np)):
      ufunc = getattr(np, member)
      if isinstance(ufunc, np.ufunc):
        self.register_ufunc(ufunc, "__call__",
                            GenericCallUfuncEmitter("numpy." + member))
    # Register generic 1-result array funcs.
    GENERIC_FUNCS = (
        (np.inner, "numpy.inner"),
        (np.outer, "numpy.outer"),
        (np.dot, "numpy.dot"),
        (np.vdot, "numpy.vdot"),
        (np.linalg.det, "numpy.linalg.det"),
        # TODO: This needs a custom implementation to differentiate when
        # axes is specified (this version will fail).
        (np.transpose, "numpy.transpose"),
    )
    for f, op_name in GENERIC_FUNCS:
      self.register_array_func(f, GenericArrayFuncEmitter(op_name))


if __name__ == "__main__":
  import doctest
  doctest.testmod()
