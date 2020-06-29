#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Base classes and interfaces."""

from collections import namedtuple
from enum import Enum
import sys
from typing import List, Optional, Sequence, Union

from _npcomp.mlir import ir
from .target import *

__all__ = [
    "Configuration",
    "Environment",
    "NameReference",
    "NameResolver",
    "PartialEvalHook",
    "PartialEvalType",
    "PartialEvalResult",
    "LiveValueRef",
    "ValueCoder",
    "ValueCoderChain",
]

_NotImplementedType = type(NotImplemented)

################################################################################
# Name resolution
################################################################################


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
  __slots__ = []

  def checked_resolve_name(self, name: str) -> Optional[NameReference]:
    ref = self.resolve_name(name)
    assert ref is not None, "Lookup of name {} is required".format(name)
    return ref

  def resolve_name(self, name: str) -> Optional[NameReference]:
    return None


################################################################################
# Value coding
# Transforms python values into IR values.
################################################################################


class ValueCoder:
  """Encodes values in various ways.

  Instances are designed to be daisy-chained and should ignore types that they
  don't understand. Functions return NotImplemented if they cannot handle a
  case locally.
  """
  __slots__ = []

  def code_py_value_as_const(self, env: "Environment",
                             py_value) -> Union[_NotImplementedType, ir.Value]:
    return NotImplemented


class ValueCoderChain(ValueCoder):
  """Codes values by delegating to sub-coders in order."""
  __slots__ = ["_sub_coders"]

  def __init__(self, sub_coders: Sequence[ValueCoder]):
    self._sub_coders = tuple(sub_coders)

  def __repr__(self):
    return "ValueCoderChain({})".format(self._sub_coders)

  def code_py_value_as_const(self, env: "Environment",
                             py_value) -> Union[_NotImplementedType, ir.Value]:
    for sc in self._sub_coders:
      result = sc.code_py_value_as_const(env, py_value)
      if result is not NotImplemented:
        return result
    return NotImplemented


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

  def as_partial_eval_result(self) -> "PartialEvalResult":
    return self

  @staticmethod
  def not_evaluated() -> "PartialEvalResult":
    return PartialEvalResult(PartialEvalType.NOT_EVALUATED, NotImplemented)

  @staticmethod
  def yields_live_value(live_value) -> "PartialEvalResult":
    assert isinstance(live_value, LiveValueRef)
    return PartialEvalResult(PartialEvalType.YIELDS_LIVE_VALUE, live_value)

  @staticmethod
  def yields_ir_value(ir_value: ir.Value) -> "PartialEvalResult":
    assert isinstance(ir_value, ir.Value)
    return PartialEvalResult(PartialEvalType.YIELDS_IR_VALUE, ir_value)

  @staticmethod
  def error() -> "PartialEvalResult":
    return PartialEvalResult(PartialEvalType.ERROR, sys.exc_info())

  @staticmethod
  def error_message(message: str) -> "PartialEvalResult":
    try:
      raise RuntimeError(message)
    except RuntimeError:
      return PartialEvalResult.error()


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

  def as_partial_eval_result(self) -> PartialEvalResult:
    return PartialEvalResult.yields_live_value(self)

  def resolve_getattr(self, env: "Environment",
                      attr_name: str) -> PartialEvalResult:
    """Gets a named attribute from the live value."""
    return PartialEvalResult.not_evaluated()

  def resolve_call(self, env: "Environment", args: Sequence[ir.Value],
                   keywords: Sequence[str]) -> PartialEvalResult:
    """Resolves a function call given 'args' and 'keywords'."""
    return PartialEvalResult.not_evaluated()

  def __repr__(self):
    return "LiveValueRef({}, {})".format(self.__class__.__name__,
                                         self.live_value)


class PartialEvalHook:
  """Hook interface for performing partial evaluation."""
  __slots__ = []

  def partial_evaluate(self, py_value) -> PartialEvalResult:
    raise NotImplementedError


################################################################################
# Configuration and environment
################################################################################


class Configuration:
  """Base class providing global configuration objects."""
  __slots__ = [
      "target_factory",
      "base_name_resolvers",
      "value_coder",
      "partial_eval_hook",
  ]

  def __init__(self,
               *,
               target_factory: TargetFactory,
               base_name_resolvers: Sequence[NameResolver] = (),
               value_coder: Optional[ValueCoder] = None,
               partial_eval_hook: PartialEvalHook = None):
    super().__init__()
    self.target_factory = target_factory
    self.base_name_resolvers = tuple(base_name_resolvers)
    self.value_coder = value_coder if value_coder else ValueCoderChain(())
    self.partial_eval_hook = partial_eval_hook

  def __repr__(self):
    return ("Configuration(target_factory={}, base_name_resolvers={}, "
            "value_code={}, partial_eval_hook={})").format(
                self.target_factory, self.base_name_resolvers, self.value_coder,
                self.partial_eval_hook)


class Environment:
  """Instantiated configuration for emitting code in a specific context.

  This brings together:
    - The code generation context (ir_h)
    - An instantiated target
    - Delegating interfaces for other configuration objects.

  Note that this class does not actually implement most of the delegate
  interfaces because it hides the fact that some may require more obtuse
  APIs than should be exposed to end callers (i.e. expecting environment or
  other config objects).
  """
  __slots__ = [
      "config",
      "ir_h",
      "_name_resolvers",
      "target",
  ]

  def __init__(self,
               *,
               config: Configuration,
               ir_h: ir.DialectHelper,
               name_resolvers: Sequence[NameResolver] = ()):
    super().__init__()
    self.config = config
    self.ir_h = ir_h
    self.target = config.target_factory(self.ir_h)
    self._name_resolvers = (tuple(name_resolvers) +
                            self.config.base_name_resolvers)

  def resolve_name(self, name: str) -> Optional[NameReference]:
    for resolver in self._name_resolvers:
      ref = resolver.resolve_name(name)
      if ref is not None:
        return ref
    return None

  def partial_evaluate(self, py_value) -> PartialEvalResult:
    return self.config.partial_eval_hook.partial_evaluate(py_value)

  def code_py_value_as_const(self,
                             py_value) -> Union[_NotImplementedType, ir.Value]:
    return self.config.value_coder.code_py_value_as_const(self, py_value)
