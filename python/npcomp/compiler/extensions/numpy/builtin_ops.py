#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Configures evaluation support for numpy builtin ops."""

from typing import Callable, Iterator, Sequence, Tuple

import functools
import numpy as np

from _npcomp.mlir import ir

from ... import logging
from ...interfaces import *
from ...partial_eval_base import *

__all__ = [
    "get_ufuncs_from_module",
    "bind_ufuncs",
]

################################################################################
# Ufunc evaluation
################################################################################


def _default_ufunc_predicate(ufunc: np.ufunc) -> bool:
  """Filters ufuncs based on ability to evaluate them."""
  # Support up to 2 input, 1 output functions.
  if ufunc.nin > 2 or ufunc.nout != 1:
    return False
  return True


def get_ufuncs_from_module(
    *,
    module=np,
    prefix: str = "numpy.",
    predicate: Callable[[np.ufunc], bool] = _default_ufunc_predicate,
) -> Iterator[Tuple[str, np.ufunc]]:
  """Iterates over all ufuncs in a module.

  Yields:
    Tuple of (prefixed_name, ufunc).
  """
  ufunc_class = np.ufunc
  for local_name in dir(module):
    value = getattr(module, local_name)
    if isinstance(value, ufunc_class):
      if not predicate(value):
        logging.debug("Skipped ufunc: {}{} ({})", prefix, local_name, value)
      else:
        yield (prefix + local_name), value


def bind_ufuncs(ufuncs: Iterator[Tuple[str, np.ufunc]],
                pe_hook: MappedPartialEvalHook):
  """Binds a set of ufuncs to a partial eval hook."""
  for qualified_name, ufunc in ufuncs:
    pe_hook.bind_action(functools.partial(BuiltinUfuncLiveValueRef,
                                          qualified_name, ufunc),
                        for_ref=ufunc)


class BuiltinUfuncLiveValueRef(LiveValueRef):
  """A partial evaluation that emits IR for invoking a ufunc."""
  __slots__ = ["_qualified_name", "_ufunc"]

  def __init__(self, qualified_name: str, ufunc: np.ufunc, live_value):
    super().__init__(live_value)
    self._qualified_name = qualified_name
    self._ufunc = ufunc

  def resolve_call(self, env: Environment, args: Sequence[ir.Value],
                   keywords: Sequence[str]) -> PartialEvalResult:
    if keywords:
      return PartialEvalResult.error_message(
          "ufunc call does not currently support keyword args")
    if len(args) != self._ufunc.nin:
      return PartialEvalResult.error_message(
          "ufunc {} expected {} inputs but got {}".format(
              self._qualified_name, self._ufunc.nin, len(args)))
    ir_h = env.ir_h
    # Because a ufunc call is defined in terms of tensors and, at this stage,
    # all "public" values are ndarray, do appropriate conversions.
    tensor_args = [ir_h.numpy_copy_to_tensor_op(arg).result for arg in args]
    result_type = ir_h.numpy_unknown_tensor_type
    tensor_result = ir_h.numpy_builtin_ufunc_call_op(
        *tensor_args,
        qualified_name=self._qualified_name,
        result_type=result_type).result
    array_result = ir_h.numpy_create_array_from_tensor_op(tensor_result).result
    return PartialEvalResult.yields_ir_value(array_result)
