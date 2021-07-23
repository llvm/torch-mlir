#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

_refjit = None

BACKEND_PASSES = (
    "func(convert-scf-to-std)",
    "func(canonicalize)",
    "func(tcf-shape-refinement)",
)


def get_refjit():
  """Dynamically resolves the refjit backend native module."""
  global _refjit
  if _refjit is not None:
    return _refjit
  from .... import _cext
  try:
    imported_refjit = _cext.backend.refjit
  except AttributeError:
    raise ImportError(
        "The npcomp native module was not compiled with refjit support")
  _refjit = imported_refjit
  return _refjit


def is_enabled() -> bool:
  """Returns whether the backend is enabled for the current build."""
  try:
    _get_refjit()
    return True
  except ImportError:
    return False


def get_runtime_libs():
  # The _refjit_resources directory is at the npcomp.compiler level.
  resources_dir = os.path.join(os.path.dirname(__file__))
  return [os.path.join(resources_dir, "libNPCOMPCompilerRuntimeShlib.so")]


class JitModuleInvoker:
  """Wrapper around a native JitModule for calling functions."""

  def __init__(self, jit_module):
    super().__init__()
    self._jit_module = jit_module

  def __getattr__(self, function_name):
    return self.__getitem__(function_name)

  def __getitem__(self, function_name):

    def invoke(*args):
      results = self._jit_module.invoke(function_name, args)
      if len(results) == 1:
        # De-tuple.
        return results[0]
      else:
        return tuple(results)

    invoke.__isnpcomp__ = True
    return invoke
