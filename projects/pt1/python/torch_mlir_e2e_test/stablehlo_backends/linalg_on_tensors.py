# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from torch_mlir import ir
from torch_mlir.ir import *
from torch_mlir.passmanager import *
from torch_mlir.compiler_utils import run_pipeline_with_repro_report

from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import (
    RefBackendLinalgOnTensorsBackend,
)

from .abc import StablehloBackend

from torch_mlir._mlir_libs._stablehlo import eval_module
import numpy as np

__all__ = [
    "LinalgOnTensorsStablehloBackend",
]

element_type_to_np_dtype = {
    "i1": np.bool_,
    "i8": np.int8,
    "ui8": np.uint8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
}

def convert_dense_elements_attr_to_numpy(attr):
    dense_attr = ir.DenseElementsAttr(attr)
    # print(dense_attr.type)
    # print(dense_attr.type.shape)
    # print(dense_attr.type.element_type)
    generic_format = str(dense_attr)
    array, shape = generic_format.split(":")
    obj = eval(array.split("<")[1].split(">")[0])
    dtype = element_type_to_np_dtype[str(dense_attr.type.element_type)]
    if isinstance(obj, list):
        return np.array(obj, dtype=dtype)
    else:
        return np.full(dense_attr.type.shape, obj, dtype=dtype)

class RefBackendInvoker:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, function_name: str):
        def invoke(*args):
            mlir_args = [ir.DenseElementsAttr.get(arg, context=self.module.context) for arg in args]
            rets = eval_module(self.module, mlir_args)
            rets = [convert_dense_elements_attr_to_numpy(i) for i in rets]
            if len(rets) == 1:
                return rets[0]
            return rets
        return invoke

# The pipeline of func.func passes that lower the STABLEHLO backend contract to the
# Linalg-on-Tensors backend contract accepted by RefBackend.
STABLEHLO_TO_LINALG_FUNC_PIPELINE = ",".join(
    [
        "func.func(stablehlo-aggressive-simplification)",
        "stablehlo-legalize-to-linalg",
        "stablehlo-convert-to-signless",
        "canonicalize",
    ]
)

SHAPE_LEGALIZE_TO_STABLEHLO_PIPELINE = ",".join(
    [
        "func.func(shape-legalize-to-stablehlo)",
        "canonicalize",
    ]
)

def raise_if_not_supported_ops(module: Module):
    for func in module.body.operations:
        assert isinstance(func, ir.dialects.func.FuncOp)
        for op in func.entry_block.operations:
            if not op.operation.name.startswith("stablehlo."):
                raise RuntimeError(f"stablehlo interpreter doesn't support {op.operation.name}")
            if op.operation.name == "stablehlo.batch_norm_inference":
                raise RuntimeError(f"stablehlo interpreter doesn't support {op.operation.name}")

class LinalgOnTensorsStablehloBackend(StablehloBackend):
    """Main entry-point for the linalg-on-tensors based Stablehlo backend.

    This currently uses the linalg-on-tensors RefBackend for actual execution.
    """

    def __init__(self):
        super().__init__()
        self.refbackend = RefBackendLinalgOnTensorsBackend()

    def compile(self, imported_module: Module):
        """Compiles an imported module that satisfied the Stablehlo backend contract.

        Args:
          imported_module: The MLIR module consisting of funcs in the Stablehlo dialect.
        Returns:
          An opaque, backend specific compiled artifact object that can be
          passed to `load`.
        """
        with imported_module.context:
            # print(imported_module.operation.get_asm())
            copied_module = Module.parse(imported_module.operation.get_asm())
        try:
            run_pipeline_with_repro_report(
                imported_module,
                f"builtin.module({STABLEHLO_TO_LINALG_FUNC_PIPELINE})",
                "Lowering STABLEHLO backend contract to Linalg-on-Tensors backend contract",
            )
            result = self.refbackend.compile(imported_module)
            return (result, "linalg")
        except:
            pass

        run_pipeline_with_repro_report(
            copied_module,
            f"builtin.module({SHAPE_LEGALIZE_TO_STABLEHLO_PIPELINE})",
            "shape-legalize-to-stablehlo",
        )
        raise_if_not_supported_ops(copied_module)
        # print(copied_module)
        return (copied_module, "stablehlo")

    def load(self, module):
        """Loads a compiled artifact into the runtime."""
        if module[1] == "linalg":
            return self.refbackend.load(module[0])
        else:
            return RefBackendInvoker(module[0])