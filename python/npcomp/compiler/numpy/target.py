#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import *

from ..utils.mlir_utils import *

from ... import ir as _ir

__all__ = [
    "GenericTarget32",
    "GenericTarget64",
    "Target",
    "TargetFactory",
]


class Target:
  """
  Abstract class providing configuration and hooks for a specific compilation
  target.
  """
  __slots__ = [
      "ic",
  ]

  def __init__(self, ic):
    self.ic = ic

  @property
  def target_name(self) -> str:
    return NotImplementedError()

  @property
  def impl_int_type(self) -> _ir.Type:
    """Gets the default int type for the backend for the Python 'int' type."""
    raise NotImplementedError()

  @property
  def impl_float_type(self) -> _ir.Type:
    """Gets the implementation's type for the python 'float' type."""
    raise NotImplementedError()


class GenericTarget64(Target):
  """A generic 64 bit target."""

  @property
  def target_name(self) -> str:
    return "generic64"

  @property
  def impl_int_type(self) -> _ir.Type:
    """Gets the default int type for the backend for the Python 'int' type."""
    return _ir.IntegerType.get_signless(64, context=self.ic.context)

  @property
  def impl_float_type(self) -> _ir.Type:
    """Gets the implementation's type for the python 'float' type."""
    return _ir.F64Type.get(context=self.ic.context)


class GenericTarget32(Target):
  """A generic 32 bit target (uses 32bit ints and floats)."""

  @property
  def target_name(self) -> str:
    return "generic32"

  @property
  def impl_int_type(self) -> _ir.Type:
    """Gets the default int type for the backend for the Python 'int' type."""
    return _ir.IntegerType.get_signless(32, context=self.ic.context)

  @property
  def impl_float_type(self) -> _ir.Type:
    """Gets the implementation's type for the python 'float' type."""
    return _ir.F32Type.get(context=self.ic.context)


# Factory for producing a target (matches the Target constructor).
TargetFactory = Callable[[ImportContext], Target]
