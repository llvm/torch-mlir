# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.compiler as ireec
import iree.runtime as ireert
from torch_mlir_e2e_test.linalg_on_tensors_backends.abc import LinalgOnTensorsBackend
from torch_mlir_e2e_test.torchscript.configs import LinalgOnTensorsBackendTestConfig


class IREEInvoker:
    def __init__(self, iree_module):
        self._iree_module = iree_module

    def __getattr__(self, function_name: str):
        def invoke(*args):
            return self._iree_module[function_name](*args)

        return invoke


class IREELinalgOnTensorsBackend(LinalgOnTensorsBackend):
    """Main entry-point for the reference backend."""

    def __init__(self):
        super().__init__()

    def compile(self, imported_module):
        """Compiles an imported module, with a flat list of functions.
        The module is expected to be in linalg-on-tensors + scalar code form.
        TODO: More clearly define the backend contract. Generally this will
        extend to support globals, lists, and other stuff.

        Args:
          imported_module: The MLIR module consisting of funcs in the torch
            dialect.
        Returns:
          An opaque, backend specific compiled artifact object that can be
          passed to `load`.
        """
        return ireec.compile_str(str(imported_module),
                                 target_backends=["dylib-llvm-aot"])

    def load(self, flatbuffer) -> IREEInvoker:
        """Loads a compiled artifact into the runtime."""
        vm_module = ireert.VmModule.from_flatbuffer(flatbuffer)
        config = ireert.Config(driver_name="dylib")
        ctx = ireert.SystemContext(config=config)
        ctx.add_vm_module(vm_module)
        return IREEInvoker(ctx.modules.module)


config = LinalgOnTensorsBackendTestConfig(IREELinalgOnTensorsBackend())
COMMON_TORCH_MLIR_LOWERING_XFAILS = {
    "QuantizedMLP_basic",
    "TableBatchEmbeddingModule_basic",
    "MobilenetV2Module_basic",
    "MobilenetV3Module_basic",
}
xfail_set = COMMON_TORCH_MLIR_LOWERING_XFAILS
