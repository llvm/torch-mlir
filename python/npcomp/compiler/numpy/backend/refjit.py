#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

from _npcomp import mlir
from npcomp.compiler.utils import logging

__all__ = [
    "is_enabled",
    "CompilerBackend",
]

FRONTEND_PASSES = (
    "npcomp-cpa-type-inference",
    "numpy-public-functions-to-tensor",
    "convert-numpy-to-tcf",
    "convert-scf-to-std",
    "canonicalize",
    "tcf-shape-refinement",
)

_refjit = None


def _get_refjit():
  """Dynamically resolves the refjit backend native module."""
  global _refjit
  if _refjit is not None:
    return _refjit
  try:
    from _npcomp.backend import refjit as imported_refjit
  except ImportError:
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
  resources_dir = os.path.join(os.path.dirname(__file__), "..", "..",
                               "_refjit_resources")
  return [os.path.join(resources_dir, "libNPCOMPCompilerRuntimeShlib.so")]


class CompilerBackend:
  """Main entry-point for the backend."""

  def __init__(self):
    super().__init__()
    self._refjit = _get_refjit()
    self._debug = logging.debug_enabled()

  def compile(self, imported_ir_module: mlir.ir.ModuleOp):
    """Compiles an imported module.

    Args:
      imported_ir_module: The MLIR module as imported from the ImportFrontend.
    Returns:
      An opaque, backend specific module object that can be passed to load.
      The object may actually be something more specific to the backend (i.e.
      for IREE, it is a serialized VM flatbuffer) but the contract is that
      it is operated on by methods on this class.
    """
    # Frontend.
    pm = mlir.passes.PassManager(imported_ir_module.context)
    pm.addPassPipelines(*FRONTEND_PASSES)
    pm.run(imported_ir_module)
    if self._debug:
      logging.debug("Frontend IR:{}", imported_ir_module.to_asm())

    # Backend.
    # Note that this is a separate pass manager purely to aid in debugging.
    pm = mlir.passes.PassManager(imported_ir_module.context)
    self._refjit.build_backend_compilation_pipeline(pm)
    pm.run(imported_ir_module)
    if self._debug:
      logging.debug("Backend IR:{}", imported_ir_module.to_asm())

    jit_module = self._refjit.JITModule.from_compiled_module(
        imported_ir_module, get_runtime_libs())
    return jit_module

  def load(self, jit_module):
    """Loads a compiled artifact into the runtime.

    Since this is a JIT instead of an AOT compiler,
    """
    return JitModuleInvoker(jit_module)


class JitModuleInvoker:
  """Wrapper around a native JitModule for calling functions."""

  def __init__(self, jit_module):
    super().__init__()
    self._jit_module = jit_module

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
