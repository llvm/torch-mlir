#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

from mlir.ir import *
from mlir.passmanager import *
from npcomp.compiler.generic.backend import refjit as refjit_backend
from npcomp.compiler.utils import logging

__all__ = [
    "is_enabled",
    "CompilerBackend",
]

FRONTEND_PASSES = (
    "builtin.func(npcomp-cpa-type-inference)",
    "numpy-public-functions-to-tensor",
    "builtin.func(convert-scf-to-std)",
    "builtin.func(canonicalize)",
)

# Re-export.
is_enabled = refjit_backend.is_enabled


class CompilerBackend:
  """Main entry-point for the backend."""

  def __init__(self):
    super().__init__()
    self._refjit = refjit_backend.get_refjit()
    self._debug = logging.debug_enabled()

  def compile(self, imported_module: Module):
    """Compiles an imported module.

    Args:
      legacy_imported_ir_module: The MLIR module as imported from the
        ImportFrontend.
    Returns:
      An opaque, backend specific module object that can be passed to load.
      The object may actually be something more specific to the backend (i.e.
      for IREE, it is a serialized VM flatbuffer) but the contract is that
      it is operated on by methods on this class.
    """
    with imported_module.context as context:
      # Frontend.
      if self._debug:
        logging.debug("Input IR:\n{}", imported_module)
      assert (
          imported_module.operation.verify()), "Imported module does not verify"
      pm = PassManager.parse(",".join(FRONTEND_PASSES))
      pm.run(imported_module)
      if self._debug:
        logging.debug("Frontend IR:\n{}", imported_module)

      # Backend.
      # Note that this is a separate pass manager purely to aid in debugging.
      pm = PassManager()
      self._refjit.build_backend_compilation_pipeline(pm)
      pm.run(imported_module)
      if self._debug:
        logging.debug("Backend IR:\n{}", imported_module)

    jit_module = self._refjit.JITModule.from_compiled_module(
        imported_module, refjit_backend.get_runtime_libs())
    return jit_module

  def load(self, jit_module):
    """Loads a compiled artifact into the runtime.

    Since this is a JIT instead of an AOT compiler,
    """
    return refjit_backend.JitModuleInvoker(jit_module)
