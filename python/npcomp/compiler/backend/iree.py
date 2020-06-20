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
    "basicpy-type-inference",
    "convert-basicpy-to-std",
    "canonicalize",
    "convert-scf-to-std",
)

_ireec = None
_ireert = None
_cached_config = None


def _get_iree():
  """Dynamically resolves the iree backend module."""
  global _ireec
  global _ireert
  if _ireec is not None:
    return _ireec, _ireert
  try:
    from _npcomp.backend import iree as imported_ireec
  except ImportError:
    raise ImportError(
        "The npcomp native module was not compiled with IREE support")
  try:
    from pyiree import rt as imported_rt
  except ImportError:
    raise ImportError("IREE runtime library not found (pyiree.rt)")

  _ireec = imported_ireec
  _ireert = imported_rt
  return _ireec, _ireert


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
    self._ireec, self._ireert = _get_iree()
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
    ireec = self._ireec
    # For easier debugging, split into to pass manager invocations.
    # Frontend.
    pm = mlir.passes.PassManager(imported_ir_module.context)
    self.add_frontend_passes(pm)
    pm.run(imported_ir_module)
    if self._debug:
      logging.debug("Frontend IR:{}", imported_ir_module.to_asm())
    # Backend.
    pm = mlir.passes.PassManager(imported_ir_module.context)
    self.add_backend_passes(pm)
    pm.run(imported_ir_module)
    if self._debug:
      logging.debug("Backend IR:{}", imported_ir_module.to_asm())
    # Translation/serialization.
    vm_blob = ireec.translate_to_vm_bytecode(imported_ir_module)
    if self._debug:
      logging.debug("Compiled VM BLOB size={}", len(vm_blob))
    return vm_blob

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

  def add_frontend_passes(self, pm: mlir.passes.PassManager):
    """Adds passes needed for legalizing from an imported form.

    While an arbitrary distinction, the passes added here are more about
    legalizing the basicpy and numpy dialects in preparation for performing
    backend compilation. They are separated to aid debugging.
    """
    # TOOD: Have an API for this
    pm.addPassPipelines(*FRONTEND_PASSES)

  def add_backend_passes(self, pm: mlir.passes.PassManager):
    """Adds passes for full backend compilation.

    These passes are added after the frontend passes.
    """
    ireec = self._ireec
    ireec.build_flow_transform_pass_pipeline(pm)
    ireec.build_hal_transform_pass_pipeline(pm)
    ireec.build_vm_transform_pass_pipeline(pm)
