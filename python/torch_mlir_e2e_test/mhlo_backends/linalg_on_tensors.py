# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from torch_mlir.ir import *
from torch_mlir.passmanager import *
from torch_mlir.compiler_utils import run_pipeline_with_repro_report

from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend

from .abc import MhloBackend

__all__ = [
    "LinalgOnTensorsMhloBackend",
]

class LinalgOnTensorsMhloBackend(MhloBackend):
    """Main entry-point for the linalg-on-tensors based MHLO backend.

    This currently uses the linalg-on-tensors RefBackend for actual execution.
    """
    def __init__(self):
        super().__init__()
        self.refbackend = RefBackendLinalgOnTensorsBackend()

    def compile(self, imported_module: Module):
        """Compiles an imported module that satisfied the MHLO backend contract.

        Args:
          imported_module: The MLIR module consisting of funcs in the MHLO
            dialect.
        Returns:
          An opaque, backend specific compiled artifact object that can be
          passed to `load`.
        """
        run_pipeline_with_repro_report(
            imported_module,
            "builtin.module(func.func(symbolic-shape-optimization),func.func(hlo-legalize-to-linalg),func.func(canonicalize))",
            "Lowering MLIR-HLO to Linalg-on-Tensors")
        return self.refbackend.compile(imported_module)

    def load(self, module):
        """Loads a compiled artifact into the runtime."""
        return self.refbackend.load(module)
