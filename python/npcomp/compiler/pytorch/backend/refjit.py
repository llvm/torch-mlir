#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

import torch

from mlir.ir import *
from mlir.passmanager import *
from npcomp.compiler.generic.backend import refjit as refjit_backend
from npcomp.compiler.utils import logging

__all__ = [
    "is_enabled",
    "CompilerBackend",
]

TORCH_TO_TCF_PASSES = (
    "func(aten-recognize-kernels)",
    "func(convert-aten-to-tcf)",
    "numpy-public-functions-to-tensor",
    "canonicalize",
)

# Re-export.
is_enabled = refjit_backend.is_enabled


class TorchJitModuleInvoker(refjit_backend.JitModuleInvoker):
  """Allows torch.Tensor inputs to be passed to module invocations."""

  def __getitem__(self, function_name: str):
    numpy_invoke = super().__getitem__(function_name)

    def invoke(*args):
      args = tuple(
          arg.numpy() if isinstance(arg, torch.Tensor) else arg for arg in args)
      return numpy_invoke(*args)

    return invoke


class CompilerBackend:
  """Main entry-point for the backend."""

  def __init__(self):
    super().__init__()
    self._refjit = refjit_backend.get_refjit()
    self._debug = logging.debug_enabled()

  def compile(self, imported_module: Module):
    """Compiles an imported module.

    Args:
      imported_module: The MLIR module consisting of funcs in the torch
        dialect.
    Returns:
      An opaque, backend specific module object that can be passed to load.
      The object may actually be something more specific to the backend (i.e.
      for IREE, it is a serialized VM flatbuffer) but the contract is that
      it is operated on by methods on this class.
    """
    # TODO: Once transitioned to new Python API, don't reparse the module.
    with Context() as context:
      if self._debug:
        logging.debug("Initial PyTorch IR:\n{}", imported_module)

      # Frontend.
      pm = PassManager.parse(",".join(TORCH_TO_TCF_PASSES))
      pm.run(imported_module)
      if self._debug:
        logging.debug("TCF IR:\n{}", imported_module)

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

  def load(self, jit_module) -> TorchJitModuleInvoker:
    """Loads a compiled artifact into the runtime."""
    return TorchJitModuleInvoker(jit_module)
