# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from torch_mlir.ir import *
from torch_mlir.passmanager import *
from torch_mlir.compiler_utils import run_pipeline_with_repro_report

from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend

from .abc import TosaBackend

__all__ = [
    "LinalgOnTensorsTosaBackend",
]

class LinalgOnTensorsTosaBackend(TosaBackend):
    """Main entry-point for the linalg-on-tensors based TOSA backend.

    This currently uses the linalg-on-tensors RefBackend for actual execution.
    """
    def __init__(self):
        super().__init__()
        self.refbackend = RefBackendLinalgOnTensorsBackend()

    def compile(self, imported_module: Module):
        """Compiles an imported module that satisfied the TOSA backend contract.

        Args:
          imported_module: The MLIR module consisting of funcs in the TOSA
            dialect.
        Returns:
          An opaque, backend specific compiled artifact object that can be
          passed to `load`.
        """

        # TOSA legalization may emit tosa.const() ops. These are legalized
        # by tosa-to-arith to arith.constants. This mechanical transformation 
        # must be done prior to TOSA-to-LinAlg so that the latter does not fail. 
        # This is an artifact of legalizations spread across a collection of simple
        # ones in TOSA-to-Standard and the main conversions TOSA-to-LinAlg, 
        # that depend on TOSA as well as TOSA-to-Standard.  
        run_pipeline_with_repro_report(
            imported_module,
            "builtin.module(func.func(tosa-to-arith))",
            "Lowering TOSA to Arith")

        # Named ops must be legalized prior to general tosa-to-linalg
        run_pipeline_with_repro_report(
            imported_module,
            "builtin.module(func.func(tosa-to-linalg-named))",
            "Lowering TOSA to Linalg-on-Tensors for Named Ops")

        # TOSA-to-LinAlg may generate tosa.const() ops, so we want to lower them
        # to arith.constants here before proceeding further.
        run_pipeline_with_repro_report(
            imported_module,
            "builtin.module(func.func(tosa-to-tensor),func.func(tosa-to-linalg),func.func(tosa-to-arith))",
            "Lowering TOSA to Linalg-on-Tensors")

        return self.refbackend.compile(imported_module)

    def load(self, module):
        """Loads a compiled artifact into the runtime."""
        return self.refbackend.load(module)
