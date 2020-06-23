#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from typing import Optional, Union

from _npcomp.mlir import ir

from . import logging
from .target import *

__all__ = [
    "BuiltinsValueCoder",
    "Environment",
    "NameReference",
    "NameResolver",
    "ValueCoder",
    "ValueCoderChain",
]


class ValueCoder:
  """Encodes values in various ways.

  Instances are designed to be daisy-chained and should ignore types that they
  don't understand. Functions return NotImplemented if they cannot handle a
  case locally.
  """
  __slots__ = []

  def create_const(self, env: "Environment", py_value):
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


class NameReference:
  """Abstract base class for performing operations on a name."""
  __slots__ = [
      "name",
  ]

  def __init__(self, name):
    super().__init__()
    self.name = name

  def load(self, env: "Environment",
           ir_h: ir.DialectHelper) -> Optional[ir.Value]:
    """Loads the IR Value associated with the name.

    The load may either be direct, returning an existing value or
    side-effecting, causing a read from an external context.

    Args:
      ir_h: The dialect helper used to emit code.
    Returns:
      An SSA value containing the resolved value (or None if not bound).
    Raises:
      NotImplementedError if load is not supported for this name.
    """
    raise NotImplementedError()

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


class Environment(NameResolver):
  """Manages access to the environment of a code region.

  This encapsulates name lookup, access to the containing module, etc.
  """
  __slots__ = [
      "ir_h",
      "_name_resolvers",
      "target",
      "value_coder",
  ]

  def __init__(self,
               ir_h: ir.DialectHelper,
               *,
               target: Target,
               name_resolvers=(),
               value_coder):
    super().__init__()
    self.ir_h = ir_h
    self.target = target
    self._name_resolvers = name_resolvers
    self.value_coder = value_coder

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
    except AttributeError:
      assert False, "Function {} does not have a __code__ attribute".format(f)

    # Locals resolver.
    # Note that co_varnames should include both parameter and local names.
    locals_resolver = LocalNameResolver(code.co_varnames)
    resolvers = (locals_resolver,)
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


class LocalNameReference(NameReference):
  """Holds an association between a name and SSA value."""
  __slots__ = [
      "_current_value",
  ]

  def __init__(self, name, initial_value=None):
    super().__init__(name)
    self._current_value = initial_value

  def load(self, env: "Environment") -> Optional[ir.Value]:
    return self._current_value

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
