#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re

import numpy as np

from . import context
from ..native.mlir import edsc


def _map_typing_to_mlir_type(mlir_m, typing_annot):
  """Maps a typing annotation to an MLIR type.

  Args:
    mlir_m: MLIRModule.
    typing_annot: Value for an __annotations__ entry.
  Returns:
    MLIR type or None if not mappable.
  """
  if typing_annot is np.ndarray:
    return mlir_m.make_type("tensor<*x!numpy.any_dtype>")
  return None


class GenericFunctionTrace:
  """Represents a trace of a 'generic' python function in progress."""

  def __init__(self, mlir_m, mlir_f):
    self._mlir_m = mlir_m
    self._mlir_f = mlir_f

  @property
  def mlir_module(self):
    return self._mlir_m

  @property
  def mlir_function(self):
    return self._mlir_f

  @classmethod
  def from_typed_pyfunc(cls, mlir_m, pyfunc, name_in_module=None):
    """Creates a generic function trace from a pyfunc with type annotations.

    This is a relatively limited mechanism which relies on typing annotations
    for arguments and results and supports a relatively limited amount of
    variation.

    Examples:

    * Generic ndarrays:
      >>> m = edsc.MLIRModule()
      >>> def simple_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
      ...   return a * b
      >>> gft = GenericFunctionTrace.from_typed_pyfunc(m, simple_mul)
      >>> ir = gft.mlir_module.get_ir()
      >>> print(re.findall("func @simple_mul.+", ir)[0])
      func @simple_mul$$generic(%arg0: tensor<*x!numpy.any_dtype> {py_name = "a"}, %arg1: tensor<*x!numpy.any_dtype> {py_name = "b"}) -> tensor<*x!numpy.any_dtype> attributes {py_ftype = "generic_trace", py_name = "simple_mul"} {

    * None types must be annotated:
      >>> m = edsc.MLIRModule()
      >>> def simple_mul(a: np.ndarray, b: np.ndarray) -> None:
      ...   return a * b
      >>> gft = GenericFunctionTrace.from_typed_pyfunc(m, simple_mul)
      >>> ir = gft.mlir_module.get_ir()
      >>> print(re.findall("func @simple_mul.+", ir)[0])
      func @simple_mul$$generic(%arg0: tensor<*x!numpy.any_dtype> {py_name = "a"}, %arg1: tensor<*x!numpy.any_dtype> {py_name = "b"}) attributes {py_ftype = "generic_trace", py_name = "simple_mul"} {

    Args:
      mlir_m: An MLIRModule.
      pyfunc: A python function to transform.
    Returns:
      A new GenericFunctionTrace.
    """
    if name_in_module is None:
      name_in_module = pyfunc.__name__ + "$$generic"
    code = pyfunc.__code__
    # Process arguments.
    f_args = []
    for i in range(code.co_argcount):
      arg_name = code.co_varnames[i]
      arg_annot = pyfunc.__annotations__.get(arg_name)
      if arg_annot is None:
        raise ValueError("Function %s arg %d is missing a typing annotation" % (
            pyfunc.__name__, i))
      arg_type = _map_typing_to_mlir_type(mlir_m, arg_annot)
      if arg_type is None:
        raise ValueError("Function %s arg %d is not a supported type" % (
            pyfunc.__name__, i))     
      arg_type = arg_type({
          "py_name": mlir_m.stringAttr(arg_name),
      })
      f_args.append(arg_type)

    # Process results.
    f_results = []
    if "return" not in pyfunc.__annotations__:
      raise ValueError("Un-annotated function returns not yet supported")
    return_annot = pyfunc.__annotations__["return"]
    if return_annot is not None:
      return_type = _map_typing_to_mlir_type(mlir_m, return_annot)
      if return_type is None:
        raise ValueError("Function %s return type %r is not supported" % (
            pyfunc.__name__, return_annot))
      f_results.append(return_type)

    mlir_f = mlir_m.make_function(
        name_in_module, f_args, f_results,
        py_ftype=mlir_m.stringAttr("generic_trace"),
        py_name=mlir_m.stringAttr(pyfunc.__name__))
    return GenericFunctionTrace(mlir_m, mlir_f)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
