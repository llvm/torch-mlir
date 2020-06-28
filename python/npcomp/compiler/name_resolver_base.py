#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Name resolvers for common scenarios."""

from typing import Optional

from _npcomp.mlir import ir

from .interfaces import *

__all__ = [
    "ConstModuleNameResolver",
    "LocalNameResolver",
]

################################################################################
# Local name resolution
# This is used for local names that can be managed purely as SSA values.
################################################################################


class LocalNameReference(NameReference):
  """Holds an association between a name and SSA value."""
  __slots__ = [
      "_current_value",
  ]

  def __init__(self, name, initial_value=None):
    super().__init__(name)
    self._current_value = initial_value

  def load(self, env: Environment) -> PartialEvalResult:
    if self._current_value is None:
      return PartialEvalResult.error_message(
          "Attempt to access local '{}' before assignment".format(self.name))
    return PartialEvalResult.yields_ir_value(self._current_value)

  def store(self, env: Environment, value: ir.Value):
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

  def resolve_name(self, name) -> Optional[NameReference]:
    return self._name_refs.get(name)


################################################################################
# Constant name resolution
# For some DSLs, it can be appropriate to treat some containing scopes as
# constants. This strategy typically binds to a module and routes loads
# through the partial evaluation hook.
################################################################################


class ConstNameReference(NameReference):
  """Represents a name/value mapping that will emit as a constant."""
  __slots__ = [
      "_py_value",
  ]

  def __init__(self, name, py_value):
    super().__init__(name)
    self._py_value = py_value

  def load(self, env: Environment) -> PartialEvalResult:
    return env.partial_evaluate(self._py_value)

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

  def resolve_name(self, name) -> Optional[NameReference]:
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
