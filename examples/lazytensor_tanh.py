# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""
Example of taking a Lazy Tensor computation and compiling it using torch-mlir.

This example depends on the Lazy Tensor Core (LTC) of PyTorch. For information
on how to obtain LTC, see here:
https://github.com/pytorch/pytorch/blob/lazy_tensor_staging/lazy_tensor_core/QUICKSTART.md

To run the example, make sure the following are in your PYTHONPATH:
    1. /path/to/torch-mlir/examples
    2. /path/to/pytorch/lazy_tensor_core
    3. /path/to/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir

then, simply call `python lazytensor_tanh.py`.
"""

import numpy as np
import torch
import lazy_tensor_core as ltc
from torch._C import CompilationUnit

from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend \
    import RefBackendLinalgOnTensorsBackend
from torch_mlir.passmanager import PassManager

from utils.annotator import Annotation
from utils.torch_mlir_types import TorchTensorType
from lazytensor.builder import build_module

ltc._LAZYC._ltc_init_ts_backend()
device = 'lazy'
dtype = torch.float32
shape = (2, 3)

x = torch.randn(shape, device=device, dtype=dtype)
y = torch.randn(shape, device=device, dtype=dtype)

def computation(x, y):
    return y * x.tanh()

# Capture lazy computation and convert to TorchScript IR
graph_str = ltc._LAZYC._get_ltc_tensors_backend([computation(x, y)])
print("LAZY GRAPH")
print(graph_str)
graph = torch._C.parse_ir(graph_str)

# Create a torch.jit.ScriptFunction out of the graph
cu = CompilationUnit()
func_name = 'my_method'
script_function = cu.create_function(func_name, graph)

# `build_module` takes the torch.jit.ScriptFunction and the
# annotation on the operand types, and outputs an `ir.Module`
# with a single function representing the ScriptFunction in
# the torch MLIR dialect
func_annotation = Annotation([TorchTensorType(shape=shape, dtype=torch.float),
                              TorchTensorType(shape=shape, dtype=torch.float)])
mlir_module = build_module(script_function, func_annotation)

print("MLIR")
mlir_module.dump()

# Compile the torch MLIR and execute the compiled program
with mlir_module.context:
    pm = PassManager.parse('torch-function-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline')
pm.run(mlir_module)

print("BEFORE LINALG-ON-TENSORS BACKEND PIPELINE")
print(mlir_module)

backend = RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(mlir_module)
jit_module = backend.load(compiled)

print("\n\nRunning Example Calculation")
print("Compiled result:")
print(jit_module.my_method(x.cpu().numpy(), y.cpu().numpy()))
print("Expected result:")
print(computation(x, y))
