#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import namedtuple
from enum import Enum
import inspect
import sys
from typing import Optional, Union

from _npcomp.mlir import ir

from . import logging
from .py_value_utils import *
from .target import *

__all__ = [
    "BuiltinsValueCoder",
    "Environment",
    "LiveValueRef",
    "NameReference",
    "NameResolver",
    "PartialEvalResult",
    "PartialEvalType",
    "PartialEvalHook",
    "ResolveAttrLiveValueRef",
    "ValueCoder",
    "ValueCoderChain",
]

_Unspec = object()

################################################################################
# Interfaces and base classes
################################################################################


class ValueCoder:
  """Encodes values in various ways.

  Instances are designed to be daisy-chained and should ignore types that they
  don't understand. Functions return NotImplemented if they cannot handle a
  case locally.
  """
  __slots__ = []

  def create_const(self, env: "Environment", py_value):
    return NotImplemented


class NameReference:
  """Abstract base class for performing operations on a name."""
  __slots__ = [
      "name",
  ]

  def __init__(self, name):
    super().__init__()
    self.name = name

  def load(self, env: "Environment",
           ir_h: ir.DialectHelper) -> "PartialEvalResult":
    """Loads the IR Value associated with the name.

    The load may either be direct, returning an existing value or
    side-effecting, causing a read from an external context.

    Args:
      ir_h: The dialect helper used to emit code.
    Returns:
      A partial evaluation result.
    """
    return PartialEvalResult.not_evaluated()

  def store(self, env: "Environment", value: ir.Value, ir_h: ir.DialectHelper):
    """Stores a new value into the name.

    A subsequent call to 'load' should yield the same value, subject to
    typing constraints on value equality.

    Args:
      value: The new value to store into the name.
      ir_h: The dialect helper used to emit code.
    Raises:
      NotImplementedError if store is not supported for this name.
    """
    raise NotImplementedError()


class NameResolver:
  """Abstract base class that can resolve a name.

  Name resolvers are typically stacked.
  """

  def checked_lookup(self, name):
    ref = self.lookup(name)
    assert ref is not None, "Lookup of name {} is required".format(name)
    return ref

  def lookup(self, name) -> Optional[NameReference]:
    return None


################################################################################
# Partial evaluation
# When the compiler is extracting from a running program, it is likely that
# evaluations produce live values which can be further partially evaluated
# at import time, in the context of the running instance (versus emitting
# program IR to do so). This behavior is controlled through a PartialEvalHook
# on the environment.
################################################################################


class PartialEvalType(Enum):
  # Could not be evaluated immediately and the operation should be
  # code-generated. yields NotImplemented.
  NOT_EVALUATED = 0

  # Yields a LiveValueRef
  YIELDS_LIVE_VALUE = 1

  # Yields an IR value
  YIELDS_IR_VALUE = 2

  # Evaluation yielded an error (yields contains exc_info from sys.exc_info()).
  ERROR = 3


class PartialEvalResult(namedtuple("PartialEvalResult", "type,yields")):
  """Encapsulates the result of a partial evaluation."""

  @classmethod
  def not_evaluated(cls):
    return cls(PartialEvalType.NOT_EVALUATED, NotImplemented)

  @classmethod
  def yields_live_value(cls, live_value):
    assert isinstance(live_value, LiveValueRef)
    return cls(PartialEvalType.YIELDS_LIVE_VALUE, live_value)

  @classmethod
  def yields_ir_value(cls, ir_value):
    assert isinstance(ir_value, ir.Value)
    return cls(PartialEvalType.YIELDS_IR_VALUE, ir_value)

  @classmethod
  def error(cls):
    return cls(PartialEvalType.ERROR, sys.exc_info())

  @classmethod
  def error_message(cls, message):
    try:
      raise RuntimeError(message)
    except RuntimeError:
      return cls.error()


class LiveValueRef:
  """Wraps a live value from the containing environment.

  Typically, when expressions encounter a live value, a limited number of
  partial evaluations can be done against it in place (versus emitting the code
  to import it and perform the operation). This default base class will not
  perform any static evaluations.
  """
  __slots__ = [
      "live_value",
  ]

  def __init__(self, live_value):
    super().__init__()
    self.live_value = live_value

  def resolve_getattr(self, env: "Environment", attr_name) -> PartialEvalResult:
    """Gets a named attribute from the live value."""
    return PartialEvalResult.not_evaluated()

  def resolve_call(self, env: "Environment", args,
                   keywords) -> PartialEvalResult:
    """Resolves a function call given 'args' and 'keywords'."""
    return PartialEvalResult.not_evaluated()

  def __repr__(self):
    return "MacroValueRef({}, {})".format(self.__class__.__name__,
                                          self.live_value)


class ResolveAttrLiveValueRef(LiveValueRef):
  """Custom MacroValueRef that will resolve attributes via getattr."""
  __slots__ = []

  def resolve_getattr(self, env: "Environment", attr_name) -> PartialEvalResult:
    logging.debug("RESOLVE_GETATTR '{}' on {}".format(attr_name,
                                                      self.live_value))
    try:
      attr_py_value = getattr(self.live_value, attr_name)
    except:
      return PartialEvalResult.error()
    return env.partial_eval_hook.resolve(attr_py_value)


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


class PartialEvalHook:
  """Owned by an environment to customize partial evaluation."""
  __slots__ = [
      "_value_map",
  ]

  def __init__(self):
    super().__init__()
    self._value_map = PyValueMap()

  def resolve(self, py_value) -> PartialEvalResult:
    """Performs partial evaluation on a python value."""
    binding = self._value_map.lookup(py_value)
    if binding is None:
      logging.debug("PARTIAL EVAL RESOLVE {}: Passthrough", py_value)
      return PartialEvalResult.yields_live_value(LiveValueRef(py_value))
    if isinstance(binding, LiveValueRef):
      logging.debug("PARTIAL EVAL RESOLVE {}: {}", py_value, binding)
      return PartialEvalResult.yields_live_value(binding)
    if isinstance(binding, PartialEvalResult):
      return binding
    # Attempt to call.
    try:
      binding = binding(py_value)
      assert isinstance(binding, PartialEvalResult), (
          "Expected PartialEvalResult but got {}".format(binding))
      logging.debug("PARTIAL EVAL RESOLVE {}: {}", py_value, binding)
      return binding
    except:
      return PartialEvalResult.error()

  def _bind(self,
            binding,
            *,
            for_ref=_Unspec,
            for_type=_Unspec,
            for_predicate=_Unspec):
    if for_ref is not _Unspec:
      self._value_map.bind_reference(for_ref, binding)
    elif for_type is not _Unspec:
      self._value_map.bind_type(for_type, binding)
    elif for_predicate is not _Unspec:
      self._value_map.bind_predicate(for_predicate, binding)
    else:
      raise ValueError(
          "Must specify one of 'for_ref', 'for_type' or 'for_predicate")

  def enable_getattr(self, **kwargs):
    """Enables partial evaluation of getattr."""
    self._bind(
        lambda pv: PartialEvalResult.yields_live_value(
            ResolveAttrLiveValueRef(pv)), **kwargs)

  def enable_template_call(self, callee_name, **kwargs):
    """"Enables a global template call."""
    self._bind(
        lambda pv: PartialEvalResult.yields_live_value(
            TemplateCallLiveValueRef(callee_name, pv)), **kwargs)


################################################################################
# Environment
# Top level instance encapsulating access to runtime state.
################################################################################


class Environment(NameResolver):
  """Manages access to the environment of a code region.

  This encapsulates name lookup, access to the containing module, etc.
  """
  __slots__ = [
      "ir_h",
      "_name_resolvers",
      "target",
      "value_coder",
      "partial_eval_hook",
  ]

  def __init__(self,
               ir_h: ir.DialectHelper,
               *,
               target: Target,
               name_resolvers=(),
               value_coder,
               partial_eval_hook=None):
    super().__init__()
    self.ir_h = ir_h
    self.target = target
    self._name_resolvers = name_resolvers
    self.value_coder = value_coder
    self.partial_eval_hook = partial_eval_hook if partial_eval_hook else PartialEvalHook(
    )

  @classmethod
  def for_const_global_function(cls, ir_h: ir.DialectHelper, f, *,
                                parameter_bindings, **kwargs):
    """Helper to generate an environment for a global function.

    This is a helper for the very common case and will be wholly insufficient
    for advanced cases, including mutable global state, closures, etc.
    Globals from the module are considered immutable.
    """
    try:
      code = f.__code__
      globals_dict = f.__globals__
      builtins_module = globals_dict["__builtins__"]
    except AttributeError:
      assert False, (
          "Function {} does not have required user-defined function attributes".
          format(f))

    # Locals resolver.
    # Note that co_varnames should include both parameter and local names.
    locals_resolver = LocalNameResolver(code.co_varnames)
    resolvers = (
        locals_resolver,
        ConstModuleNameResolver(globals_dict, as_dict=True),
        ConstModuleNameResolver(builtins_module),
    )
    env = cls(ir_h, name_resolvers=resolvers, **kwargs)

    # Bind parameters.
    for name, value in parameter_bindings:
      logging.debug("STORE PARAM: {} <- {}", name, value)
      locals_resolver.checked_lookup(name).store(env, value)
    return env

  def lookup(self, name) -> Optional[NameReference]:
    for resolver in self._name_resolvers:
      ref = resolver.lookup(name)
      if ref is not None:
        return ref
    return None


################################################################################
# Standard name resolvers
################################################################################


class LocalNameReference(NameReference):
  """Holds an association between a name and SSA value."""
  __slots__ = [
      "_current_value",
  ]

  def __init__(self, name, initial_value=None):
    super().__init__(name)
    self._current_value = initial_value

  def load(self, env: "Environment") -> PartialEvalResult:
    if self._current_value is None:
      return PartialEvalResult.error_message(
          "Attempt to access local '{}' before assignment".format(self.name))
    return PartialEvalResult.yields_ir_value(self._current_value)

  def store(self, env: "Environment", value: ir.Value):
    self._current_value = value

  def __repr__(self):
    return "<LocalNameReference({})>".format(self.name)


class LocalNameResolver(NameResolver):
  """Resolves names in a local cache of SSA values.

  This is used to manage locals and arguments (that are not referenced through
  a closure).
  """
  __slots__ = [
      "_name_refs",
  ]

  def __init__(self, names):
    super().__init__()
    self._name_refs = {name: LocalNameReference(name) for name in names}

  def lookup(self, name) -> Optional[NameReference]:
    return self._name_refs.get(name)


class ConstNameReference(NameReference):
  """Represents a name/value mapping that will emit as a constant."""
  __slots__ = [
      "_py_value",
  ]

  def __init__(self, name, py_value):
    super().__init__(name)
    self._py_value = py_value

  def load(self, env: "Environment") -> PartialEvalResult:
    return env.partial_eval_hook.resolve(self._py_value)

  def __repr__(self):
    return "<ConstNameReference({}={})>".format(self.name, self._py_value)


class ConstModuleNameResolver(NameResolver):
  """Resolves names from a module by treating them as immutable and loading
  them as constants into a function scope.
  """
  __slots__ = [
      "_as_dict",
      "module",
  ]

  def __init__(self, module, *, as_dict=False):
    super().__init__()
    self.module = module
    self._as_dict = as_dict

  def lookup(self, name) -> Optional[NameReference]:
    if self._as_dict:
      if name in self.module:
        py_value = self.module[name]
      else:
        return None
    else:
      try:
        py_value = getattr(self.module, name)
      except AttributeError:
        return None
    return ConstNameReference(name, py_value)


################################################################################
# Standard value coders
################################################################################


class BuiltinsValueCoder:
  """Value coder for builtin python types."""
  __slots__ = []

  def create_const(self, env: "Environment", py_value):
    ir_h = env.ir_h
    ir_c = ir_h.context
    if py_value is True:
      return ir_h.basicpy_bool_constant_op(True).result
    elif py_value is False:
      return ir_h.basicpy_bool_constant_op(False).result
    elif py_value is None:
      return ir_h.basicpy_singleton_op(ir_h.basicpy_NoneType).result
    elif isinstance(py_value, int):
      ir_type = env.target.impl_int_type
      ir_attr = ir_c.integer_attr(ir_type, py_value)
      return ir_h.constant_op(ir_type, ir_attr).result
    elif isinstance(py_value, float):
      ir_type = env.target.impl_float_type
      ir_attr = ir_c.float_attr(ir_type, py_value)
      return ir_h.constant_op(ir_type, ir_attr).result
    elif isinstance(py_value, str):
      return ir_h.basicpy_str_constant_op(py_value).result
    elif isinstance(py_value, bytes):
      return ir_h.basicpy_bytes_constant_op(py_value).result
    elif isinstance(py_value, type(...)):
      return ir_h.basicpy_singleton_op(ir_h.basicpy_EllipsisType).result
    return NotImplemented


class ValueCoderChain(ValueCoder):
  """Codes values by delegating to sub-coders in order."""
  __slots__ = ["_sub_coders"]

  def __init__(self, sub_coders):
    self._sub_coders = sub_coders

  def create_const(self, env: "Environment", py_value):
    for sc in self._sub_coders:
      result = sc.create_const(env, py_value)
      if result is not NotImplemented:
        return result
    return NotImplemented
