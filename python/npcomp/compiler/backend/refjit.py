#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from _npcomp import mlir
from npcomp.compiler import logging

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

    jit_module = self._refjit.JITModule.from_mlir(imported_ir_module, [])
    return jit_module

  def load(self, jit_module):
    """Loads a compiled artifact into the runtime.

    Since this is a JIT instead of an AOT compiler,
    """
