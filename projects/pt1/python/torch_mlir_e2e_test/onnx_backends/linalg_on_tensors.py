# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.


from torch_mlir.compiler_utils import run_pipeline_with_repro_report
from torch_mlir.ir import *
from torch_mlir.passmanager import *
from torch_mlir.torchscript import OutputType
from torch_mlir.torchscript import _lower_mlir_module

from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend

from .abc import OnnxBackend

__all__ = [
    "LinalgOnTensorsOnnxBackend",
]

# The pipeline of func.func passes that lower the ONNX backend contract to the
# Linalg-on-Tensors backend contract accepted by RefBackend.
ONNX_TO_TORCH_FUNC_PIPELINE = ",".join([
    "convert-torch-onnx-to-torch",
])


class LinalgOnTensorsOnnxBackend(OnnxBackend):
    """Main entry-point for the linalg-on-tensors based ONNX backend.

    This currently uses the linalg-on-tensors RefBackend for actual execution.
    """

    def __init__(self):
        super().__init__()
        self.refbackend = RefBackendLinalgOnTensorsBackend()

    def compile(self, imported_module: Module):
        """Compiles an imported module that satisfied the ONNX backend contract.

        Args:
          imported_module: The MLIR module consisting of ONNX operations wrapped by
          torch.operator.
        Returns:
          An opaque, backend specific compiled artifact object that can be
          passed to `load`.
        """
        run_pipeline_with_repro_report(
            imported_module,
            f"builtin.module(func.func({ONNX_TO_TORCH_FUNC_PIPELINE}))",
            "Lowering Onnx backend contract to Linalg-on-Tensors backend contract")

        run_pipeline_with_repro_report(
            imported_module,
            f"builtin.module(torch-lower-to-backend-contract)",
            "Lowering TorchFX IR -> Torch Backend IR",
        )

        imported_module = _lower_mlir_module(False, OutputType.LINALG_ON_TENSORS, imported_module)
        compiled_module = self.refbackend.compile(imported_module)
        return compiled_module

    def load(self, module):
        """Loads a compiled artifact into the runtime."""
        return self.refbackend.load(module)
