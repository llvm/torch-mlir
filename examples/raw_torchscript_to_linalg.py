# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""
Example of taking TorchScript as a string and compiling it using torch-mlir.
This is useful for testing lowering of Torch ops that don't have a Python
binding.

To run the example, make sure the following are in your PYTHONPATH:
    1. /path/to/torch-mlir/examples
    2. /path/to/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir

then, simply call `python raw_torchscript_to_linalg.py`.
"""

import numpy as np
import torch
from torch._C import CompilationUnit

from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend \
    import RefBackendLinalgOnTensorsBackend
from torch_mlir.passmanager import PassManager

from utils.annotator import Annotation
from utils.torch_mlir_types import TorchTensorType
from lazytensor.builder import build_module

def _print_title(title: str):
    print()
    print(title)
    print('-' * len(title))

graph_str = """\
graph(%p0 : Tensor, %p1 : Tensor):
  %0 : int = prim::Constant[value=1]()
  %1 : Tensor  = aten::add(%p0, %p1, %0)
  return (%1)
"""

graph = torch._C.parse_ir(graph_str)
_print_title('TorchScript Graph')
print(graph)

# Create a torch.jit.ScriptFunction out of the graph
cu = CompilationUnit()
func_name = 'my_method'
script_function = cu.create_function(func_name, graph)

# `build_module` takes the torch.jit.ScriptFunction and the
# annotation on the operand types, and outputs an `ir.Module`
# with a single function representing the ScriptFunction in
# the torch MLIR dialect
input_shape = (3, 4)
input_dtype = torch.float
func_annotation = Annotation([TorchTensorType(shape=input_shape, dtype=input_dtype),
                              TorchTensorType(shape=input_shape, dtype=input_dtype)])
mlir_module = build_module(script_function, func_annotation)

_print_title("Torch-MLIR")
mlir_module.dump()

# Compile the torch MLIR and execute the compiled program
with mlir_module.context:
    pipeline = ','.join(['torch-function-to-torch-backend-pipeline',
                         'torch-backend-to-linalg-on-tensors-backend-pipeline'])
    pm = PassManager.parse(pipeline)
pm.run(mlir_module)

_print_title("Linalg-MLIR")
print(mlir_module)

backend = RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(mlir_module)
jit_module = backend.load(compiled)

_print_title("Running Compiled Graph")
x = torch.randn(input_shape, dtype=input_dtype)
y = torch.randn(input_shape, dtype=input_dtype)
print('Expected output:')
print(script_function(x, y))
print('Output from compiled MLIR:')
print(jit_module.my_method(x.numpy(), y.numpy()))
