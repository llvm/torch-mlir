# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

from typing import List

import torch
import torch.nn as nn
from torch.export import Dim
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_compiler, get_aot_graph_name, set_model_name

from torch_mlir import fx


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# CHECK-LABEL: test_import_frozen_exported_program
# CHECK:     func.func @main(%[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32>
# CHECK-DAG: %[[a:.+]] = torch.vtensor.literal(dense_resource<torch_tensor_1_4_torch.float32> : tensor<1x4xf32>) : !torch.vtensor<[1,4],f32>
# CHECK-DAG: %[[b:.+]] = torch.vtensor.literal(dense_resource<torch_tensor_3_1_torch.float32> : tensor<3x1xf32>) : !torch.vtensor<[3,1],f32>
# CHECK-DAG: %[[p:.+]] = torch.vtensor.literal(dense<{{.*>+}} : tensor<1x1xf32>) : !torch.vtensor<[1,1],f32>
# CHECK-DAG: %[[tanh:.+]] = torch.aten.tanh %[[ARG0]]
# CHECK-DAG: %[[mul_a:.+]] = torch.aten.mul.Tensor %[[tanh]], %[[a]]
# CHECK-DAG: %[[mul_b:.+]] = torch.aten.mul.Tensor %[[mul_a]], %[[b]]
# CHECK-DAG: %[[mul_p:.+]] = torch.aten.mul.Tensor %[[mul_b]], %[[p]]
# CHECK:     return %[[mul_p]]
#
# Validate dialect resources exist.
# CHECK: dialect_resources:
# CHECK-DAG: torch_tensor_1_4_torch.float32
# CHECK-DAG: torch_tensor_3_1_torch.float32
def test_import_frozen_exported_program():
    # Tests the basic structural premises of import_frozen_exported_program,
    # namely that free tensors (buffers) and parameters are treated as
    # literals and frozen.
    @torch._dynamo.assume_constant_result
    def get_a():
        return torch.randn(1, 4)

    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = torch.randn(3, 1)
            self.p = nn.Parameter(torch.randn(1, 1))

        def forward(self, x):
            return torch.tanh(x) * get_a() * self.b * self.p

    m = fx.export_and_import(Basic(), torch.randn(3, 4))
    print(m)


@run
# CHECK-LABEL: test_import_frozen_exported_program_with_func_name
# CHECK:     func.func @test_net(%[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32>
def test_import_frozen_exported_program_with_func_name():
    @torch._dynamo.assume_constant_result
    def get_a():
        return torch.randn(1, 4)

    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = torch.randn(3, 1)
            self.p = nn.Parameter(torch.randn(1, 1))

        def forward(self, x):
            return torch.tanh(x) * get_a() * self.b * self.p

    m = fx.export_and_import(Basic(), torch.randn(3, 4), func_name="test_net")
    print(m)

@run
# CHECK-LABEL: test_import_frozen_exported_program_with_dynamic_shapes
# CHECK:     func.func @test_net(%[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[?,4],f32>) -> !torch.vtensor<[?,4],f32>
def test_import_frozen_exported_program_with_dynamic_shapes():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.tanh(x)

    batch = Dim("batch")
    dynamic_shapes = {"x": {0: batch}}
    m = fx.export_and_import(Basic(), torch.randn(3, 4), dynamic_shapes=dynamic_shapes, func_name="test_net")
    print(m)



@make_boxed_compiler
def fx_import_aot_autograd_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(gm.print_readable(False), flush=True)
    m = fx.stateless_fx_import(gm, model_name=get_aot_graph_name())
    print(m, flush=True)
    return gm

@run
# CHECK-LABEL: test_stateless_fx_import
# CHECK:     func.func @basic_forward__6_inference_0(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32>
# CHECK-NEXT:  %0 = torch.aten.tanh %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
# CHECK-NEXT:  return %0 : !torch.vtensor<[3,4],f32>
def test_stateless_fx_import():
    fx_import_backend = aot_autograd(fw_compiler=fx_import_aot_autograd_backend)
    set_model_name("basic_forward")
    @torch._dynamo.optimize(backend=fx_import_backend)
    def basic_forward(x):
        return torch.tanh(x)

    basic_forward(torch.randn(3, 4))
