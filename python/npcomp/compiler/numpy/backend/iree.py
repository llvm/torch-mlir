#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
import subprocess

from mlir.ir import *
from mlir.passmanager import *
from _npcomp import register_dialects
from _npcomp import mlir as legacy_mlir
from npcomp.compiler.generic.backend import iree as iree_backend
from npcomp.compiler.utils import logging

__all__ = [
    "is_enabled",
    "CompilerBackend",
]

FRONTEND_PASSES = (
    "func(basicpy-type-inference)",
    "func(convert-basicpy-to-std)",
    "func(canonicalize)",
    "func(convert-scf-to-std)",
)

_ireert = None
_cached_config = None


def _get_iree():
  """Dynamically resolves the iree backend module."""
  global _ireert
  try:
    from pyiree import rt as imported_rt
  except ImportError:
    raise ImportError("IREE runtime library not found (pyiree.rt)")
  _ireert = imported_rt
  return _ireert


def is_enabled() -> bool:
  """Returns whether the backend is enabled for the current build."""
  try:
    _get_iree()
    return True
  except ImportError:
    return False


class CompilerBackend:
  """Main entry-point for the backend."""

  def __init__(self):
    super().__init__()
    self._ireert = _get_iree()
    self._debug = logging.debug_enabled()

  def compile(self, legacy_imported_ir_module: legacy_mlir.ir.ModuleOp):
    """Compiles an imported module.

    Args:
      imported_ir_module: The MLIR module as imported from the ImportFrontend.
    Returns:
      An opaque, backend specific module object that can be passed to load.
      The object may actually be something more specific to the backend (i.e.
      for IREE, it is a serialized VM flatbuffer) but the contract is that
      it is operated on by methods on this class.
    """
    # TODO: Once transitioned to new Python API, don't reparse the module.
    with Context() as context:
      register_dialects(context)
      imported_module = Module.parse(legacy_imported_ir_module.to_asm())
      # Frontend.
      pm = PassManager.parse(",".join(FRONTEND_PASSES))
      pm.run(imported_module)
      if self._debug:
        logging.debug("Frontend IR:{}", imported_module)

    # TODO: There should be some common utility for invoking backend processes
    # safely (and have options like saving temps, etc).
    args = [
        iree_backend.get_translate_exe(), "--iree-mlir-to-vm-bytecode-module"
    ]
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    imported_module.operation.print(binary=True,
                                    enable_debug_info=True,
                                    file=p.stdin)
    out, err = p.communicate()
    return out

  def load(self, vm_blob):
    """Loads a compiled artifact into the runtime.

    This is meant as a simple mechanism for testing and is not optimized or
    highly parameterized. It loads a compiled result into a new runtime
    instance and returns an object that exposes a python function for each
    public function compiled in the imported_ir_module that was compiled.
    """
    ireert = self._ireert
    m = ireert.VmModule.from_flatbuffer(vm_blob)
    global _cached_config
    if not _cached_config:
      # TODO: Need to make the configuration more flexible.
      _cached_config = ireert.Config(driver_name="vmla")
    ctx = ireert.SystemContext(config=_cached_config)
    ctx.add_module(m)
    # TODO: The implicit tying of the 'module' name has got to go.
    return ctx.modules.module
