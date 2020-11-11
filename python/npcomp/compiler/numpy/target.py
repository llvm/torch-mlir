#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import *
from _npcomp.mlir import ir

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
      "_mlir_helper",
  ]

  def __init__(self, mlir_helper: ir.DialectHelper):
    super().__init__()
    self._mlir_helper = mlir_helper

  @property
  def mlir_helper(self):
    return self._mlir_helper

  @property
  def mlir_context(self):
    return self._mlir_helper.context

  @property
  def target_name(self) -> str:
    return NotImplementedError()

  @property
  def impl_int_type(self) -> ir.Type:
    """Gets the default int type for the backend for the Python 'int' type."""
    raise NotImplementedError()

  @property
  def impl_float_type(self) -> ir.Type:
    """Gets the implementation's type for the python 'float' type."""
    raise NotImplementedError()


class GenericTarget64(Target):
  """A generic 64 bit target."""

  @property
  def target_name(self) -> str:
    return "generic64"

  @property
  def impl_int_type(self) -> ir.Type:
    """Gets the default int type for the backend for the Python 'int' type."""
    return self.mlir_helper.i64_type

  @property
  def impl_float_type(self) -> ir.Type:
    """Gets the implementation's type for the python 'float' type."""
    return self.mlir_helper.f64_type


class GenericTarget32(Target):
  """A generic 32 bit target (uses 32bit ints and floats)."""

  @property
  def target_name(self) -> str:
    return "generic32"

  @property
  def impl_int_type(self) -> ir.Type:
    """Gets the default int type for the backend for the Python 'int' type."""
    return self.mlir_helper.i32_type

  @property
  def impl_float_type(self) -> ir.Type:
    """Gets the implementation's type for the python 'float' type."""
    return self.mlir_helper.f32_type


# Factory for producing a target (matches the Target constructor).
TargetFactory = Callable[[ir.DialectHelper], Target]
