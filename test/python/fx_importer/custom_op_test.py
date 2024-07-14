# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch
import torch.nn as nn
from torch.export import Dim
from torch.library import Library, impl, impl_abstract

from torch_mlir import fx


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# CHECK-LABEL: test_tanh_sigmoid_cat_custom_op
# CHECK:      func.func @main(
# CHECK-SAME:       %[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[?,?,3],f32>,
# CHECK-SAME:       %[[ARG1:[a-zA-Z0-9]+]]: !torch.vtensor<[?,?,3],f32>,
# CHECK-SAME:       %[[ARG2:[a-zA-Z0-9]+]]: !torch.vtensor<[?,?,3],f32>) -> !torch.vtensor<[?,?,3],f32> {
# CHECK:        %[[S0:.+]] = torch.symbolic_int "s0" {min_val = 5, max_val = 10} : !torch.int
# CHECK:        %[[S1:.+]] = torch.symbolic_int "s1" {min_val = {{[0-9]+}}, max_val = 100} : !torch.int
# CHECK:        %[[S2:.+]] = torch.symbolic_int "s3" {min_val = {{[0-9]+}}, max_val = 50} : !torch.int
# CHECK:        %[[S3:.+]] = torch.symbolic_int "s5" {min_val = {{[0-9]+}}, max_val = {{[0-9]+}}} : !torch.int
# CHECK:        torch.bind_symbolic_shape %[[ARG0]], [%[[S0]], %[[S1]]], affine_map<()[s0, s1] -> (s0, s1, 3)> : !torch.vtensor<[?,?,3],f32>
# CHECK:        torch.bind_symbolic_shape %[[ARG1]], [%[[S0]], %[[S2]]], affine_map<()[s0, s1] -> (s0, s1, 3)> : !torch.vtensor<[?,?,3],f32>
# CHECK:        torch.bind_symbolic_shape %[[ARG2]], [%[[S0]], %[[S3]]], affine_map<()[s0, s1] -> (s0, s1, 3)> : !torch.vtensor<[?,?,3],f32>
# CHECK:        %[[OP:.+]] = torch.operator "torch.my_custom_library.tanh_sigmoid_cat_op"(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (!torch.vtensor<[?,?,3],f32>, !torch.vtensor<[?,?,3],f32>, !torch.vtensor<[?,?,3],f32>) -> !torch.vtensor<[?,?,3],f32>
# CHECK:        torch.bind_symbolic_shape %[[OP]], [%[[S0]], %[[S1]], %[[S2]], %[[S3]]], affine_map<()[s0, s1, s2, s3] -> (s0, s2 + s3 + s1 * 2, 3)> : !torch.vtensor<[?,?,3],f32>
# CHECK:        return %[[OP]] : !torch.vtensor<[?,?,3],f32>
def test_tanh_sigmoid_cat_custom_op():

    m = Library("my_custom_library", "DEF")
    m.define("tanh_sigmoid_cat_op(Tensor x, Tensor y, Tensor z) -> Tensor")

    @impl(m, "tanh_sigmoid_cat_op", "CompositeExplicitAutograd")
    def custom_op(x, y, z):
        a = torch.tanh(x)
        b = torch.sigmoid(y)
        return torch.cat((a, a, b, z), dim=1)

    @impl_abstract("my_custom_library::tanh_sigmoid_cat_op")
    def custom_op_meta(x, y, z):
        result = custom_op(x, y, z)
        return torch.empty_like(result)

    class TanhSigmoidCatCustomOp(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y, z):
            return torch.ops.my_custom_library.tanh_sigmoid_cat_op(x, y, z)

    # Sample inputs
    x = torch.randn(5, 2, 3)
    y = torch.randn(5, 6, 3)
    z = torch.randn(5, 4, 3)

    # Dynamic dim constraints
    dim_n = Dim("n", min=5, max=10)
    dim_x1 = Dim("x1", max=100)
    dim_y1 = Dim("y1", max=50)
    dim_z1 = Dim("z1")
    dynamic_shapes = {
        "x": {0: dim_n, 1: dim_x1},
        "y": {0: dim_n, 1: dim_y1},
        "z": {0: dim_n, 1: dim_z1},
    }

    m = fx.export_and_import(
        TanhSigmoidCatCustomOp(),
        x,
        y,
        z,
        dynamic_shapes=dynamic_shapes,
        import_symbolic_shape_expressions=True,
    )
    print(m)


@run
# CHECK-LABEL: test_custom_op_array_output
# CHECK:  func.func @main(%[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[?,3],f32>)
# CHECK:  %[[S0:.+]] = torch.symbolic_int "s0" {min_val = {{[0-9]+}}, max_val = 10} : !torch.int
# CHECK:  %[[int:.+]] = torch.constant.int 4
# CHECK:  %[[V0:.+]] = torch.operator "torch.my_custom_library.array_output_op"(%[[int]], %[[ARG0]]) : (!torch.int, !torch.vtensor<[?,3],f32>) -> !torch.list<vtensor>
# CHECK: %[[V1:.+]]:4 = torch.prim.ListUnpack %[[V0]] : !torch.list<vtensor> -> !torch.vtensor<[?,3],f32>, !torch.vtensor<[?,3],f32>, !torch.vtensor<[?,3],f32>, !torch.vtensor<[?,3],f32>
# CHECK: torch.bind_symbolic_shape %[[V1]]#0, [%[[S0]]], affine_map<()[s0] -> (s0, 3)> : !torch.vtensor<[?,3],f32>
# CHECK: torch.bind_symbolic_shape %[[V1]]#1, [%[[S0]]], affine_map<()[s0] -> (s0, 3)> : !torch.vtensor<[?,3],f32>
# CHECK: torch.bind_symbolic_shape %[[V1]]#2, [%[[S0]]], affine_map<()[s0] -> (s0, 3)> : !torch.vtensor<[?,3],f32>
# CHECK: torch.bind_symbolic_shape %[[V1]]#3, [%[[S0]]], affine_map<()[s0] -> (s0, 3)> : !torch.vtensor<[?,3],f32>
# CHECK: return %[[V1]]#0, %[[V1]]#1, %[[V1]]#2, %[[V1]]#3 : !torch.vtensor<[?,3],f32>, !torch.vtensor<[?,3],f32>, !torch.vtensor<[?,3],f32>, !torch.vtensor<[?,3],f32>
def test_custom_op_array_output():
    m = Library("my_custom_library", "DEF")
    m.define("array_output_op(int num_outs, Tensor a) -> Tensor[]")

    @impl(m, "array_output_op", "CompositeExplicitAutograd")
    def custom_op(num_outs, a):
        return [a] * num_outs

    @impl_abstract("my_custom_library::array_output_op")
    def custom_op_meta(num_outs, a):
        result = custom_op(num_outs, a)
        return [torch.empty_like(t) for t in result]

    class ArrayOutputCustomOp(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            return torch.ops.my_custom_library.array_output_op(4, a)

    dim = Dim("n", max=10)
    dynamic_shapes = {
        "a": {0: dim},
    }

    a = torch.rand(2, 3)
    m = fx.export_and_import(
        ArrayOutputCustomOp(),
        a,
        import_symbolic_shape_expressions=True,
        dynamic_shapes=dynamic_shapes,
    )
    print(m)
