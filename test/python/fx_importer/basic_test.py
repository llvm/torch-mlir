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
from torch._functorch.aot_autograd import (
    make_boxed_compiler,
    get_aot_graph_name,
    set_model_name,
)

from torch_mlir import fx
from torch_mlir.compiler_utils import run_pipeline_with_repro_report


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# CHECK-LABEL: test_import_frozen_exported_program
# CHECK:     func.func @main(%[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32>
# CHECK-DAG: %[[a:.+]] = torch.aten.randn
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
# CHECK:     func.func @test_net(%[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[?,?,5],f32>) -> !torch.vtensor<[?,?,5],f32>
# CHECK:     %[[S0:.*]] = torch.symbolic_int "{{[a-z0-9]+}}" {min_val = {{[0-9]+}}, max_val = {{[0-9]+}}} : !torch.int
# CHECK:     %[[S1:.*]] = torch.symbolic_int "{{[a-z0-9]+}}" {min_val = 2, max_val = {{[0-9]+}}} : !torch.int
# CHECK-DISABLED:     torch.bind_symbolic_shape %[[ARG0]], [%[[S1]], %[[S0]]], affine_map<()[s0, s1] -> (s1, s0, 5)> : !torch.vtensor<[?,?,5],f32>
# CHECK:     %[[TANH:.*]] = torch.aten.tanh %[[ARG0]] : !torch.vtensor<[?,?,5],f32> -> !torch.vtensor<[?,?,5],f32>
# CHECK-DISABLED:     torch.bind_symbolic_shape %[[TANH]], [%[[S1]], %[[S0]]], affine_map<()[s0, s1] -> (s1, s0, 5)> : !torch.vtensor<[?,?,5],f32>
# CHECK:     return %[[TANH]] : !torch.vtensor<[?,?,5],f32>
def test_import_frozen_exported_program_with_dynamic_shapes():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.tanh(x)

    batch = Dim("batch", max=10)
    channel = Dim("channel", min=2)
    dynamic_shapes = {"x": {0: batch, 1: channel}}
    m = fx.export_and_import(
        Basic(),
        torch.randn(3, 4, 5),
        dynamic_shapes=dynamic_shapes,
        func_name="test_net",
        import_symbolic_shape_expressions=True,
    )
    print(m)


@run
# CHECK-LABEL: test_broadcast_with_dynamic_shapes
# CHECK:     func.func @test_net(%[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[1,2],f32>, %[[ARG1:[a-zA-Z0-9]+]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,2],f32>
# CHECK:     %[[S0:.*]] = torch.symbolic_int "{{[a-z0-9]+}}" {min_val = {{[0-9]+}}, max_val = {{[0-9]+}}} : !torch.int
# CHECK:     torch.bind_symbolic_shape %[[ARG1]], [%[[S0]]], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],f32>
# CHECK:     torch.aten.size.int
# CHECK:     torch.prim.ListConstruct
# CHECK:     %[[EXPAND:.*]] = torch.aten.expand
# CHECK:     torch.bind_symbolic_shape %[[EXPAND]], [%[[S0]]], affine_map<()[s0] -> (s0, 2)> : !torch.vtensor<[?,2],f32>
def test_broadcast_with_dynamic_shapes():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.broadcast_to(x, (y.shape[0], -1))

    # Sample inputs
    x = torch.randn(1, 2)
    y = torch.randn(10)

    dim_0 = Dim("dim_0", max=10)
    dynamic_shapes = {
        "x": {},
        "y": {0: dim_0},
    }

    m = fx.export_and_import(
        Basic(),
        x,
        y,
        dynamic_shapes=dynamic_shapes,
        func_name="test_net",
        import_symbolic_shape_expressions=True,
    )
    print(m)


@make_boxed_compiler
def fx_import_aot_autograd_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
):
    print(gm.print_readable(False), flush=True)
    m = fx.stateless_fx_import(gm, model_name=get_aot_graph_name())
    print(m, flush=True)
    return gm


@run
# CHECK-LABEL: test_stateless_fx_import
# CHECK:     func.func @[[basic:[a-zA-Z0-9_]+]](%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32>
# CHECK-NEXT:  %0 = torch.aten.tanh %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
# CHECK-NEXT:  return %0 : !torch.vtensor<[3,4],f32>
def test_stateless_fx_import():
    fx_import_backend = aot_autograd(fw_compiler=fx_import_aot_autograd_backend)
    set_model_name("basic_forward")

    @torch._dynamo.optimize(backend=fx_import_backend)
    def basic_forward(x):
        return torch.tanh(x)

    basic_forward(torch.randn(3, 4))


@run
# CHECK-LABEL: test_full
# CHECK:    %2 = torch.aten.fill.Scalar %1, %int0 : !torch.vtensor<[],i1>, !torch.int -> !torch.vtensor<[],i1>
def test_full():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self):
            return torch.full(
                [],
                False,
                dtype=torch.bool,
                layout=torch.strided,
                device="cpu",
                pin_memory=False,
            )

    m = fx.export_and_import(Basic(), func_name="test_full", enable_graph_printing=True)
    run_pipeline_with_repro_report(
        m,
        f"builtin.module(torch-simplification-pipeline)",
        "torch-simplification-pipeline",
    )
    print(m)

@run
# CHECK-LABEL: test_while_loop_two_returns
# CHECK: func.func @test_while_loop_two_returns
# CHECK-SAME: -> (!torch.vtensor<[],si64>, !torch.vtensor<[4,4],f32>)

# Validate literal/init plumbing:
# CHECK: %[[ZERO:.*]] = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
# CHECK: %[[NONE:.*]] = torch.constant.none
# CHECK: %[[CLONE:.*]] = torch.aten.clone %[[ZERO]], %[[NONE]] : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>

# CHECK: %[[COND:.*]] = call @while_loop_cond_graph_{{[0-9]+}}(%[[CLONE]]
# CHECK: torch.aten.Bool.Tensor %[[COND]]
# CHECK: %[[MAX_ITER:.*]] = torch.constant.int 9223372036854775807
# CHECK: torch.prim.Loop %[[MAX_ITER]]

# CHECK: func.func private @while_loop_cond_graph_{{[0-9]+}}
# CHECK: torch.aten.lt.Scalar

# CHECK: func.func private @while_loop_body_graph_{{[0-9]+}}
# CHECK: torch.aten.add.Scalar
# CHECK: torch.aten.mul.Tensor
def test_while_loop_two_returns():
    class M(nn.Module):
        def forward(self, x):
            # Simple while_loop that carries a scalar and a tensor.
            def body(i, x):
                return i + 1, x * x
            i0 = torch.tensor(0)
            from torch._higher_order_ops.while_loop import while_loop

            out_i, out_x = while_loop(lambda i, x: i < 3, body, (i0, x))
            return out_i, out_x

    # Export -> import to Torch-MLIR
    m = fx.export_and_import(
        M(), torch.randn(4, 4), func_name="test_while_loop_two_returns"
    )
    print(m)

@run
# CHECK-LABEL: test_stack_trace
# CHECK: #loc[[LOC1:.+]] = loc(
# CHECK: %{{.+}} = torch.aten.add.Tensor {{.+}} loc(#loc[[LOC1]])
def test_stack_trace():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            def bar(x):
                return x + 1.0

            def foo(x, y):
                return bar(x) + bar(y)

            z = foo(x, y)
            return {"z": z}

    x = torch.randn(128, 128)
    y = torch.randn(128, 128)
    m = fx.export_and_import(Basic(), x, y, func_name="test_stack_trace")
    mlir_asm = m.operation.get_asm(enable_debug_info=True)
    print(mlir_asm)
