#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import numpy as np

from ..native.mlir import edsc
from ..exporter import *
from ..types import *


class TracingError(Exception):
  pass


class EmitterRegistry:
  def __init__(self):
    self._func_emitters = {}
    
  def register(self, func, emitter):
    self._func_emitters[func] = emitter
  
  def lookup(self, func):
    return self._func_emitters.get(func)
  
  def register_ufunc(self, ufunc, function_name):
    def emitter(pft, method, *inputs, **kwargs):
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
          op_input = pft.get_traced_array_value(py_input)
          if op_input is None:
            raise TracingError("Unregistered traced array: %r", (py_input,))
          op_inputs.append(op_input)
        
        # Emit op.  
        mlir_m = pft.mlir_module
        op_result_types = [mlir_m.make_type("tensor<*x!numpy.any_dtype>")]
        op_result = edsc.op("numpy.tmp_generic_ufunc", op_inputs, op_result_types,
                ufunc_name=mlir_m.stringAttr(function_name))
        
        # Wrap returns.
        return_array = TracedArray(pft)
        pft.set_traced_array(return_array, op_result)
        return return_array
        
      raise TracingError("Unsupported ufunc method %r:%r" % (ufunc, method,))

    self.register(ufunc, emitter)


EMITTER_REGISTRY = EmitterRegistry()
EMITTER_REGISTRY.register_ufunc(np.multiply, "numpy.multiply")
EMITTER_REGISTRY.register_ufunc(np.add, "numpy.add")


class TracedArray(np.lib.mixins.NDArrayOperatorsMixin):
  """An array that traces its operations."""
  def __init__(self, pft: "PyFuncTrace"):
    self._pft = pft
    
  def __hash__(self):
    return id(self)

  def __repr__(self):
    return "<TracedArray %d>" % id(self)

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    emitter = EMITTER_REGISTRY.lookup(ufunc)
    if emitter is None:
      return NotImplemented
    result = emitter(self._pft, method, *inputs, **kwargs)
    return result


class PyFuncTrace:
  r"""Creates an MLIR function from an unwrapped python function.
  
    # TODO: These constraints are too verbose and should be coming in by
    # example.
    >>> def simple_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ...   return a * b + a
    >>> exp = Exporter()
    >>> exp.simple_mul = simple_mul
    >>> exp.simple_mul.sig.args["a"] += Shape(1, 4)
    >>> exp.simple_mul.sig.args["a"] += DynamicDim(0)
    >>> exp.simple_mul.sig.args["a"] += DType(np.float32)
    >>> exp.simple_mul.sig.args["b"] += Shape(1)
    >>> exp.simple_mul.sig.args["b"] += DType(np.float32)
    >>> exp.simple_mul.sig.result += Shape(1, 4)
    >>> exp.simple_mul.sig.result += DynamicDim(0)
    >>> exp.simple_mul.sig.result += DType(np.float32)    
    >>> pft = PyFuncTrace(exp.simple_mul)
    >>> pft.trace()
    >>> print(pft.mlir_module.get_ir().strip())
    module {
      func @simple_mul(%arg0: tensor<?x4xf32>, %arg1: tensor<1xf32>) -> tensor<?x4xf32> {
        %0 = "numpy.tmp_generic_ufunc"(%arg0, %arg1) {ufunc_name = "numpy.multiply"} : (tensor<?x4xf32>, tensor<1xf32>) -> tensor<*x!numpy.any_dtype>
        %1 = "numpy.tmp_generic_ufunc"(%0, %arg0) {ufunc_name = "numpy.add"} : (tensor<*x!numpy.any_dtype>, tensor<?x4xf32>) -> tensor<*x!numpy.any_dtype>
        %2 = "numpy.narrow"(%1) : (tensor<*x!numpy.any_dtype>) -> tensor<?x4xf32>
        return %2 : tensor<?x4xf32>
      }
    }
  """
  __slots__ = [
    "epf",
    "mlir_ctx",
    "mlir_fun",
    "mlir_module",
    "mlir_result_types",
    "_args_array_params",
    "_traced_arrays",
    "_python_args",
    "_result_array_params",
  ]
  def __init__(self, epf: ExportPyFunction):
    self.mlir_module = edsc.MLIRModule()
    self.epf = epf
    self._traced_arrays = {}  # Mapping of TracedArray to current consumer value
    self._validate()
    
    # Extract ArrayParams for all args and results.
    self._args_array_params = [
      ArrayParams.from_constraints(arg.constraints) 
      for arg in self.epf.sig.args]
    self._python_args = [None] * len(self._args_array_params)
    self._result_array_params = ArrayParams.from_constraints(
      self.epf.sig.result.constraints)
    
    # Create the MLIR function.
    self.mlir_fun, self.mlir_result_types = self._create_mlir_function()
    self.mlir_ctx = self.mlir_module.function_context(self.mlir_fun)
    self._create_trace_roots()
    
  def set_traced_array(self, traced_array, value_handle):
    """Sets the current SSA value for a traced_array."""
    assert isinstance(traced_array, TracedArray)
    self._traced_arrays[traced_array] = value_handle

  def get_traced_array_value(self, traced_array):
    return self._traced_arrays.get(traced_array)

  def trace(self):
    # TODO: General argument merging
    with self.mlir_ctx:    
      py_results = (self.epf.pyfunc(*self._python_args),)
      if len(py_results) != len(self.mlir_result_types):
        raise TracingError(
          "Traced function returned != %d results: %r" % (
            len(self.mlir_result_types), py_results,))
        
      # Narrow all results to the declared return types.
      return_operands = []
      for py_result, mlir_result_type in zip(py_results, self.mlir_result_types):
        mlir_result = self.get_traced_array_value(py_result)
        if mlir_result is None:
          raise TracingError("Unregistered traced array: %r", (py_input,))
        # narrow to declared result type.
        return_operands.append(edsc.op(
          "numpy.narrow", [mlir_result], [mlir_result_type]))
      edsc.ret(return_operands)

  def _validate(self):
    if not all(arg.type_class == TypeClass.NdArray 
               for arg in self.epf.sig.args):
      raise NotImplementedError("Non NdArray args: %r" % (self.epf.sig.args,))
    if not self.epf.sig.result.type_class == TypeClass.NdArray:
      raise NotImplementedError("Non NdArray result: %r" % (
        self.epf.sig.result,))
  
  def _create_mlir_function(self):
    mlir_m = self.mlir_module
    epf = self.epf
    f_args = [mlir_m.make_type(ap.mlir_tensor_type_asm)
              for ap in self._args_array_params]
    f_results = [mlir_m.make_type(
      self._result_array_params.mlir_tensor_type_asm)]
    return mlir_m.make_function(epf.__name__, f_args, f_results), f_results

  def _create_trace_roots(self):
    for index, ap in enumerate(self._args_array_params):
      if ap is not None:
        ta = TracedArray(self)
        self.set_traced_array(ta, self.mlir_fun.arg(index))
        self._python_args[index] = ta
        

if __name__ == "__main__":
  import doctest
  doctest.testmod()
