#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Partial evaluation helpers and support for built-in and common scenarios."""

from typing import Any, Callable, Union

from .interfaces import *
from .py_value_utils import *
from . import logging

__all__ = [
    "MappedPartialEvalHook",
    "ResolveAttrLiveValueRef",
    "TemplateCallLiveValueRef",
]

_Unspec = object()

################################################################################
# LiveValueRef specializations for various kinds of access
################################################################################


class ResolveAttrLiveValueRef(LiveValueRef):
  """Custom LiveValueRef that will resolve attributes via getattr."""
  __slots__ = []

  def resolve_getattr(self, env: "Environment", attr_name) -> PartialEvalResult:
    logging.debug("RESOLVE_GETATTR '{}' on {}".format(attr_name,
                                                      self.live_value))
    try:
      attr_py_value = getattr(self.live_value, attr_name)
    except:
      return PartialEvalResult.error()
    return env.partial_evaluate(attr_py_value)


class TemplateCallLiveValueRef(LiveValueRef):
  """Custom LiveValueRef that resolves calls to a func_template_call op."""
  __slots__ = ["callee_name"]

  def __init__(self, callee_name, live_value):
    super().__init__(live_value)
    self.callee_name = callee_name

  def resolve_call(self, env: "Environment", args,
                   keywords) -> PartialEvalResult:
    linear_args = list(args)
    kw_arg_names = []
    for kw_name, kw_value in keywords:
      kw_arg_names.append(kw_name)
      linear_args.append(kw_value)

    ir_h = env.ir_h
    result_ir_value = ir_h.basicpy_func_template_call_op(
        result_type=ir_h.basicpy_UnknownType,
        callee_symbol=self.callee_name,
        args=linear_args,
        arg_names=kw_arg_names).result
    return PartialEvalResult.yields_ir_value(result_ir_value)


################################################################################
# PartialEvalHook implementations
################################################################################


class MappedPartialEvalHook(PartialEvalHook):
  """A PartialEvalHook that maps rules to produce live values.

  Internally, this implementation binds a predicate to an action. The predicate
  can be:
    - A python value matched by reference or value equality
    - A type that a value must be an instance of
    - An arbitrary lambda (should be limited to special cases as it forces
      a linear scan).

  An action can be one of
    - A `lambda python_value: PartialEvalResult...`
    - An object that supports as_partial_eval_result() (either a
      PartialEvalResult or LiveValueRef qualify).
    - None to indicate that the python value should be processed directly
  """
  __slots__ = [
      "_value_map",
  ]

  def __init__(self):
    super().__init__()
    self._value_map = PyValueMap()

  def __repr__(self):
    return "MappedPartialEvalHook({})".format(self._value_map)

  def partial_evaluate(self, py_value) -> PartialEvalResult:
    """Performs partial evaluation on a python value."""
    logging.debug("LOOKUP: {}", py_value)
    action = self._value_map.lookup(py_value)
    if action is None:
      # Passthrough.
      return PartialEvalResult.yields_live_value(LiveValueRef(py_value))
    # Attempt to call.
    try:
      result = action(py_value).as_partial_eval_result()
      assert isinstance(result, PartialEvalResult), (
          "Expected PartialEvalResult but got {}".format(result))
      logging.debug("PARTIAL EVAL RESOLVE {}: {}", py_value, result)
      return result
    except:
      return PartialEvalResult.error()

  def bind_action(self,
                  action: Union[PartialEvalResult, LiveValueRef,
                                Callable[[Any], PartialEvalResult]],
                  *,
                  for_ref=_Unspec,
                  for_type=_Unspec,
                  for_predicate=_Unspec):
    if hasattr(action, "as_partial_eval_result"):
      # Registers a casting action.
      action = lambda pv: pv.as_partial_eval_result()

    if for_ref is not _Unspec:
      self._value_map.bind_reference(for_ref, action)
    elif for_type is not _Unspec:
      self._value_map.bind_type(for_type, action)
    elif for_predicate is not _Unspec:
      self._value_map.bind_predicate(for_predicate, action)
    else:
      raise ValueError(
          "Must specify one of 'for_ref', 'for_type' or 'for_predicate")

  def enable_getattr(self, **kwargs):
    """Enables partial evaluation of getattr."""
    self.bind_action(
        lambda pv: PartialEvalResult.yields_live_value(
            ResolveAttrLiveValueRef(pv)), **kwargs)

  def enable_template_call(self, callee_name, **kwargs):
    """"Enables a global template call."""
    self.bind_action(
        lambda pv: PartialEvalResult.yields_live_value(
            TemplateCallLiveValueRef(callee_name, pv)), **kwargs)
