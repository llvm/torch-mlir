# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import numpy as np

from torch_mlir import ir
from torch_mlir._mlir_libs._stablehlo import eval_module

from .abc import StablehloBackend

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


class RefBackendStablehloBackend(StablehloBackend):
    """Main entry-point for Stablehlo reference backend based on Interpreter."""
    def __init__(self):
        super().__init__()

    def compile(self, imported_module: ir.Module):
        return imported_module

    def load(self, module) -> RefBackendInvoker:
        print(module.operation.get_asm())
        return RefBackendInvoker(module)
