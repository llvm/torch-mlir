# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s
# This file contains tests of various op special forms that the fx_importer
# handles.

import torch
import torch.export
import torch.nn as nn
from torch.export import Dim

from torch_mlir import fx


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# CHECK-LABEL: test_tanh_sigmoid_cat
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
# CHECK:        %[[TANH:.+]] = torch.aten.tanh %[[ARG0]] : !torch.vtensor<[?,?,3],f32> -> !torch.vtensor<[?,?,3],f32>
# CHECK:        torch.bind_symbolic_shape %[[TANH]], [%[[S0]], %[[S1]]], affine_map<()[s0, s1] -> (s0, s1, 3)> : !torch.vtensor<[?,?,3],f32>
# CHECK:        %[[SIG:.+]] = torch.aten.sigmoid %[[ARG1]] : !torch.vtensor<[?,?,3],f32> -> !torch.vtensor<[?,?,3],f32>
# CHECK:        torch.bind_symbolic_shape %[[SIG]], [%[[S0]], %[[S2]]], affine_map<()[s0, s1] -> (s0, s1, 3)> : !torch.vtensor<[?,?,3],f32>
# CHECK:        %[[LIST:.+]] = torch.prim.ListConstruct %[[TANH]], %[[TANH]], %[[SIG]], %[[ARG2]] : (!torch.vtensor<[?,?,3],f32>, !torch.vtensor<[?,?,3],f32>, !torch.vtensor<[?,?,3],f32>, !torch.vtensor<[?,?,3],f32>) -> !torch.list<vtensor>
# CHECK:        %[[CAT:.+]] = torch.aten.cat %[[LIST]], {{.*}} : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?,?,3],f32>
# CHECK:        torch.bind_symbolic_shape %[[CAT]], [%[[S0]], %[[S1]], %[[S2]], %[[S3]]], affine_map<()[s0, s1, s2, s3] -> (s0, s2 + s3 + s1 * 2, 3)> : !torch.vtensor<[?,?,3],f32>
# CHECK:        return %[[CAT]] : !torch.vtensor<[?,?,3],f32>
def test_tanh_sigmoid_cat():
    class TanhSigmoidCat(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y, z):
            a = torch.tanh(x)
            b = torch.sigmoid(y)
            return torch.cat((a, a, b, z), dim=1)

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
        TanhSigmoidCat(),
        x,
        y,
        z,
        dynamic_shapes=dynamic_shapes,
        import_symbolic_shape_expressions=True,
    )
    print(m)


@run
# CHECK-LABEL: test_symbolic_dim_differ_by_one
# CHECK:      func.func @main(%[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[?],f32>, %[[ARG1:[a-zA-Z0-9]+]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> attributes {torch.assume_strict_symbolic_shapes} {
# CHECK:        %[[S0:.+]] = torch.symbolic_int "s0" {min_val = 3, max_val = 6} : !torch.int
# This appears in torch-nightly, but not in torch-stable (re-enable once we've moved torch-stable to 2.4+)
# CHECK-DISABLED:   %[[S1:.+]] = torch.symbolic_int "s0 + 1" {min_val = 4, max_val = 7} : !torch.int
# CHECK:        torch.bind_symbolic_shape %[[ARG0]], [%[[S0]]], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],f32>
# CHECK:        torch.bind_symbolic_shape %[[ARG1]], [%[[S0]]], affine_map<()[s0] -> (s0 + 1)> : !torch.vtensor<[?],f32>
# CHECK:        %[[SLICE:.+]] = torch.aten.slice.Tensor %arg1, {{.*}}, {{.*}}, {{.*}}, {{.*}} : !torch.vtensor<[?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?],f32>
# CHECK:        torch.bind_symbolic_shape %[[SLICE]], [%[[S0]]], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],f32>
# CHECK:        %[[ADD:.+]] = torch.aten.add.Tensor %[[ARG0]], %[[SLICE]], {{.*}} : !torch.vtensor<[?],f32>, !torch.vtensor<[?],f32>, !torch.int -> !torch.vtensor<[?],f32>
# CHECK:        torch.bind_symbolic_shape %[[ADD]], [%[[S0]]], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],f32>
# CHECK:        return %[[ADD]] : !torch.vtensor<[?],f32>
def test_symbolic_dim_differ_by_one():
    class SymbolicDimDifferByOne(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x + y[1:]

    # Sample inputs
    x = torch.randn(5)
    y = torch.randn(6)

    # Dynamic dim constraints
    dimx = Dim("dimx", min=3, max=6)
    dimy = dimx + 1
    dynamic_shapes = {
        "x": {0: dimx},
        "y": {0: dimy},
    }

    m = fx.export_and_import(
        SymbolicDimDifferByOne(),
        x,
        y,
        dynamic_shapes=dynamic_shapes,
        experimental_support_mutation=True,
        import_symbolic_shape_expressions=True,
    )
    print(m)


@run
# CHECK-LABEL: test_outer_with_squared_shape
# CHECK:      func.func @main(%[[ARG0:.+]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> {
# CHECK:        %[[S0:.+]] = torch.symbolic_int "s0" {min_val = {{[0-9]+}}, max_val = {{[0-9]+}}} : !torch.int
# CHECK:        torch.bind_symbolic_shape %[[ARG0]], [%[[S0]]], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],f32>
# CHECK:        %[[VIEW1:.+]] = torch.aten.view %[[ARG0]], {{.*}} : !torch.vtensor<[?],f32>, !torch.list<int> -> !torch.vtensor<[?,1],f32>
# CHECK:        torch.bind_symbolic_shape %[[VIEW1]], [%[[S0]]], affine_map<()[s0] -> (s0, 1)> : !torch.vtensor<[?,1],f32>
# CHECK:        %[[MUL:.+]] = torch.aten.mul.Tensor %[[VIEW1]], %[[ARG0]] : !torch.vtensor<[?,1],f32>, !torch.vtensor<[?],f32> -> !torch.vtensor<[?,?],f32>
# CHECK:        torch.bind_symbolic_shape %[[MUL]], [%[[S0]]], affine_map<()[s0] -> (s0, s0)> : !torch.vtensor<[?,?],f32>
# CHECK:        %[[VIEW2:.+]] = torch.aten.view %[[MUL]], {{.*}} : !torch.vtensor<[?,?],f32>, !torch.list<int> -> !torch.vtensor<[?],f32>
# CHECK:        torch.bind_symbolic_shape %[[VIEW2]], [%[[S0]]], affine_map<()[s0] -> (s0 * s0)> : !torch.vtensor<[?],f32>
# CHECK:        return %[[VIEW2]] : !torch.vtensor<[?],f32>
def test_outer_with_squared_shape():
    class OuterWithSquaredShape(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.outer(x, x).flatten()

    # Sample inputs
    x = torch.rand(10)

    # Dynamic dim constraints
    batch = Dim("batch")
    dynamic_shapes = {"x": {0: batch}}

    m = fx.export_and_import(
        OuterWithSquaredShape(),
        x,
        dynamic_shapes=dynamic_shapes,
        import_symbolic_shape_expressions=True,
    )
    print(m)


@run
# CHECK-LABEL: test_slice_tensor_static_output
# CHECK:      func.func @main(%[[ARG0:.+]]: !torch.vtensor<[?,3],f32>) -> !torch.vtensor<[2,1],f32> {
# CHECK:        %[[S0:.+]] = torch.symbolic_int "s0" {min_val = 3, max_val = 9223372036854775806} : !torch.int
# CHECK:        torch.bind_symbolic_shape %[[ARG0]], [%[[S0]]], affine_map<()[s0] -> (s0, 3)> : !torch.vtensor<[?,3],f32>
# CHECK:        %[[SLICE1:.+]] = torch.aten.slice.Tensor %[[ARG0]], {{.*}}, {{.*}}, {{.*}}, {{.*}} : !torch.vtensor<[?,3],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,3],f32>
# CHECK:        %[[SLICE2:.+]] = torch.aten.slice.Tensor %[[SLICE1]], {{.*}}, {{.*}}, {{.*}}, {{.*}} : !torch.vtensor<[2,3],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,1],f32>
# CHECK:        return %[[SLICE2]] : !torch.vtensor<[2,1],f32>
def test_slice_tensor_static_output():
    class SliceTensorStaticOutput(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[0:2, :1]

    # Sample inputs
    x = torch.randn(4, 3)

    # Dynamic dim constraints
    batch = Dim("batch", min=3)
    dynamic_shapes = {"x": {0: batch}}

    m = fx.export_and_import(
        SliceTensorStaticOutput(),
        x,
        dynamic_shapes=dynamic_shapes,
        import_symbolic_shape_expressions=True,
    )
    print(m)


@run
# CHECK-LABEL: test_slice_tensor_dynamic_output
# CHECK:      func.func @main(%[[ARG0:.+]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> {
# CHECK:        %[[S0:.+]] = torch.symbolic_int "s0" {min_val = 5, max_val = 9223372036854775806} : !torch.int
# CHECK:        torch.bind_symbolic_shape %[[ARG0]], [%[[S0]]], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],f32>
# CHECK:        %[[SLICE:.+]] = torch.aten.slice.Tensor %[[ARG0]], {{.*}}, {{.*}}, {{.*}}, {{.*}} : !torch.vtensor<[?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?],f32>
# CHECK:        torch.bind_symbolic_shape %[[SLICE]], [%[[S0]]], affine_map<()[s0] -> (s0 - 5)> : !torch.vtensor<[?],f32>
# CHECK:        return %[[SLICE]] : !torch.vtensor<[?],f32>
def test_slice_tensor_dynamic_output():
    class SliceTensorDynamicOutput(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[5:]

    # Sample inputs
    x = torch.randn(10)

    # Dynamic dim constraints
    dimx = Dim("dimx", min=5)
    dynamic_shapes = {"x": {0: dimx}}

    m = fx.export_and_import(
        SliceTensorDynamicOutput(),
        x,
        dynamic_shapes=dynamic_shapes,
        import_symbolic_shape_expressions=True,
    )
    print(m)


@run
# CHECK-LABEL: test_div_tensor_mixed_ranks
# CHECK:      func.func @main(%[[ARG0:.+]]: !torch.vtensor<[],f32>, %[[ARG1:.+]]: !torch.vtensor<[?,3],f32>) -> !torch.vtensor<[?,3],f32> {
# CHECK:        %[[S0:.+]] = torch.symbolic_int "s0" {min_val = {{[0-9]+}}, max_val = {{[0-9]+}}} : !torch.int
# CHECK:        torch.bind_symbolic_shape %[[ARG1]], [%[[S0]]], affine_map<()[s0] -> (s0, 3)> : !torch.vtensor<[?,3],f32>
# CHECK:        %[[DIV:.+]] = torch.aten.div.Tensor %[[ARG0]], %[[ARG1]] : !torch.vtensor<[],f32>, !torch.vtensor<[?,3],f32> -> !torch.vtensor<[?,3],f32>
# CHECK:        torch.bind_symbolic_shape %[[DIV]], [%[[S0]]], affine_map<()[s0] -> (s0, 3)> : !torch.vtensor<[?,3],f32>
# CHECK:        return %[[DIV]] : !torch.vtensor<[?,3],f32>
def test_div_tensor_mixed_ranks():
    class DivTensorMixedRanks(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            div = torch.div(x, y)
            return div

    # Sample inputs
    x = torch.tensor(10.0)
    y = torch.randn(2, 3)

    # Dynamic dim constraints
    batch = Dim("batch")
    dynamic_shapes = {"x": None, "y": {0: batch}}

    m = fx.export_and_import(
        DivTensorMixedRanks(),
        x,
        y,
        dynamic_shapes=dynamic_shapes,
        import_symbolic_shape_expressions=True,
    )
    print(m)


@run
# CHECK-LABEL: test_shape_div
# CHECK:      func.func @main(%[[ARG0:.+]]: !torch.vtensor<[?,7],f32>) -> !torch.vtensor<[?,5],f32> {
# This appears in torch-nightly, but not in torch-stable (re-enable once we've moved torch-stable to 2.4+)
# CHECK-DISABLED:        %[[S0:.+]] = torch.symbolic_int "5*s1" {min_val = 0, max_val = 5000} : !torch.int
# CHECK:        %[[S1:.+]] = torch.symbolic_int "s1" {min_val = 2, max_val = 1000} : !torch.int
# CHECK:        torch.bind_symbolic_shape %[[ARG0]], [%[[S1]]], affine_map<()[s0] -> (s0 * 5, 7)> : !torch.vtensor<[?,7],f32>
# CHECK:        %[[VIEW:.+]] = torch.aten.view %[[ARG0]], {{.*}} : !torch.vtensor<[?,7],f32>, !torch.list<int> -> !torch.vtensor<[?,5],f32>
# CHECK:        torch.bind_symbolic_shape %[[VIEW]], [%[[S1]]], affine_map<()[s0] -> (s0 * 7, 5)> : !torch.vtensor<[?,5],f32>
# CHECK:        return %[[VIEW]] : !torch.vtensor<[?,5],f32>
def test_shape_div():
    class ShapeDiv(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.reshape(-1, 5)

    # Sample inputs
    x = torch.rand(10, 7)

    # Dynamic dim constraints
    batch = Dim("batch", max=1000) * 5
    dynamic_shapes = {"x": {0: batch}}

    m = fx.export_and_import(
        ShapeDiv(),
        x,
        dynamic_shapes=dynamic_shapes,
        import_symbolic_shape_expressions=True,
    )
    print(m)


@run
# CHECK-LABEL: test_broadcast_unit_dim_to_static_with_unchanged_dim_dynamic
# CHECK:      func.func @main(%[[ARG0:.+]]: !torch.vtensor<[1,?],f32>) -> !torch.vtensor<[3,?],f32> {
# CHECK:        %[[S0:.+]] = torch.symbolic_int "s0" {min_val = {{[0-9]+}}, max_val = {{[0-9]+}}} : !torch.int
# CHECK:        torch.bind_symbolic_shape %[[ARG0]], [%[[S0]]], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],f32>
# CHECK:        %[[EXPAND:.+]] = torch.aten.expand %[[ARG0]], {{.*}}, {{.*}} : !torch.vtensor<[1,?],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,?],f32>
# CHECK:        torch.bind_symbolic_shape %[[EXPAND]], [%[[S0]]], affine_map<()[s0] -> (3, s0)> : !torch.vtensor<[3,?],f32>
# CHECK:        return %[[EXPAND]] : !torch.vtensor<[3,?],f32>
def test_broadcast_unit_dim_to_static_with_unchanged_dim_dynamic():
    class BroadcastUnitDimToStaticWithUnchangedDimDynamic(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.broadcast_to(x, (3, -1))

    # Sample inputs
    x = torch.randn(1, 2)

    # Dynamic dim constraints
    dim_1 = Dim("dim_1")
    dynamic_shapes = {"x": {1: dim_1}}

    m = fx.export_and_import(
        BroadcastUnitDimToStaticWithUnchangedDimDynamic(),
        x,
        dynamic_shapes=dynamic_shapes,
        import_symbolic_shape_expressions=True,
    )
    print(m)


@run
# CHECK-LABEL: test_broadcast_unit_dim_to_dynamic_with_unchanged_dim_static
# CHECK:      func.func @main(%[[ARG0:.+]]: !torch.vtensor<[1,2],f32>, %[[ARG1:.+]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,2],f32> {
# CHECK:        %[[S0:.+]] = torch.symbolic_int "s0" {min_val = {{[0-9]+}}, max_val = {{[0-9]+}}} : !torch.int
# CHECK:        torch.bind_symbolic_shape %[[ARG1]], [%[[S0]]], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],f32>
# CHECK:        %[[EXPAND:.+]] = torch.aten.expand %[[ARG0]], {{.*}}, {{.*}} : !torch.vtensor<[1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,2],f32>
# CHECK:        torch.bind_symbolic_shape %[[EXPAND]], [%[[S0]]], affine_map<()[s0] -> (s0, 2)> : !torch.vtensor<[?,2],f32>
# CHECK:        return %3 : !torch.vtensor<[?,2],f32>
def test_broadcast_unit_dim_to_dynamic_with_unchanged_dim_static():
    class BroadcastUnitDimToDynamicWithUnchangedDimStatic(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.broadcast_to(x, (y.shape[0], -1))

    # Sample inputs
    x = torch.randn(1, 2)
    y = torch.randn(10)

    # Dynamic dim constraints
    dim_0 = Dim("dim_0")
    dynamic_shapes = {"x": {}, "y": {0: dim_0}}

    m = fx.export_and_import(
        BroadcastUnitDimToDynamicWithUnchangedDimStatic(),
        x,
        y,
        dynamic_shapes=dynamic_shapes,
        import_symbolic_shape_expressions=True,
    )
    print(m)


@run
# CHECK-LABEL: test_broadcast_unit_dim_to_dynamic_with_unchanged_dim_dynamic
# CHECK:      func.func @main(%[[ARG0:.+]]: !torch.vtensor<[1,?],f32>, %[[ARG1:.+]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?],f32> {
# CHECK:        %[[S0:.+]] = torch.symbolic_int "s0" {min_val = {{[0-9]+}}, max_val = {{[0-9]+}}} : !torch.int
# CHECK:        %[[S1:.+]] = torch.symbolic_int "s1" {min_val = {{[0-9]+}}, max_val = {{[0-9]+}}} : !torch.int
# CHECK:        torch.bind_symbolic_shape %[[ARG0]], [%[[S0]]], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],f32>
# CHECK:        torch.bind_symbolic_shape %[[ARG1]], [%[[S1]]], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],f32>
# CHECK:        %[[EXPAND:.+]] = torch.aten.expand %[[ARG0]], {{.*}}, {{.*}} : !torch.vtensor<[1,?],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,?],f32>
# CHECK:        torch.bind_symbolic_shape %[[EXPAND]], [%[[S0]], %[[S1]]], affine_map<()[s0, s1] -> (s1, s0)> : !torch.vtensor<[?,?],f32>
# CHECK:        return %[[EXPAND]] : !torch.vtensor<[?,?],f32>
def test_broadcast_unit_dim_to_dynamic_with_unchanged_dim_dynamic():
    class BroadcastUnitDimToDynamicWithUnchangedDimDynamic(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.broadcast_to(x, (y.shape[0], -1))

    # Sample inputs
    x = torch.randn(1, 2)
    y = torch.randn(10)

    # Dynamic dim constraints
    dim_0 = Dim("dim_0")
    dim_1 = Dim("dim_1")
    dynamic_shapes = {"x": {1: dim_1}, "y": {0: dim_0}}

    m = fx.export_and_import(
        BroadcastUnitDimToDynamicWithUnchangedDimDynamic(),
        x,
        y,
        dynamic_shapes=dynamic_shapes,
        import_symbolic_shape_expressions=True,
    )
    print(m)


@run
# CHECK-LABEL: test_broadcast_unit_dim_to_dynamic_with_rank_increase
# CHECK:      func.func @main(%[[ARG0:.+]]: !torch.vtensor<[1,2],f32>, %[[ARG1:.+]]: !torch.vtensor<[?,3,2],f32>) -> !torch.vtensor<[?,3,2],f32> {
# CHECK:        %[[S0:.+]] = torch.symbolic_int "s0" {min_val = {{[0-9]+}}, max_val = {{[0-9]+}}} : !torch.int
# CHECK:        torch.bind_symbolic_shape %[[ARG1]], [%[[S0]]], affine_map<()[s0] -> (s0, 3, 2)> : !torch.vtensor<[?,3,2],f32>
# CHECK:        %[[EXPAND:.+]] = torch.aten.expand %[[ARG0]], {{.*}}, {{.*}} : !torch.vtensor<[1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,3,2],f32>
# CHECK:        torch.bind_symbolic_shape %[[EXPAND]], [%[[S0]]], affine_map<()[s0] -> (s0, 3, 2)> : !torch.vtensor<[?,3,2],f32>
# CHECK:        return %[[EXPAND]] : !torch.vtensor<[?,3,2],f32>
def test_broadcast_unit_dim_to_dynamic_with_rank_increase():
    class BroadcastUnitDimToDynamicWithRankIncrease(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.broadcast_to(x, y.size())

    # Sample inputs
    x = torch.randn(1, 2)
    y = torch.randn(4, 3, 2)

    # Dynamic dim constraints
    dim_0 = Dim("dim_0")
    dynamic_shapes = {"x": {}, "y": {0: dim_0}}

    m = fx.export_and_import(
        BroadcastUnitDimToDynamicWithRankIncrease(),
        x,
        y,
        dynamic_shapes=dynamic_shapes,
        import_symbolic_shape_expressions=True,
    )
    print(m)


@run
# CHECK-LABEL: test_gather_elements
# CHECK:      func.func @main(%[[ARG0:.+]]: !torch.vtensor<[?,3],f32>, %[[ARG1:.+]]: !torch.vtensor<[2,3],si64>) -> !torch.vtensor<[2,3],f32> {
# CHECK:        %[[S0:.+]] = torch.symbolic_int "s0" {min_val = 3, max_val = 9223372036854775806} : !torch.int
# CHECK:        torch.bind_symbolic_shape %[[ARG0]], [%[[S0]]], affine_map<()[s0] -> (s0, 3)> : !torch.vtensor<[?,3],f32>
# CHECK:        %[[GATHER:.+]] = torch.aten.gather %[[ARG0]], {{.*}}, {{.*}}, {{.*}} : !torch.vtensor<[?,3],f32>, !torch.int, !torch.vtensor<[2,3],si64>, !torch.bool -> !torch.vtensor<[2,3],f32>
# CHECK:        return %[[GATHER]] : !torch.vtensor<[2,3],f32>
def test_gather_elements():
    class GatherElements(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.gather(x, 0, y)

    # Sample inputs
    x = torch.randn(4, 3)
    y = torch.tensor([[0, 0, 0], [1, 1, 1]])

    # Dynamic dim constraints
    batch = Dim("batch", min=3)
    dynamic_shapes = {"x": {0: batch}, "y": {}}

    m = fx.export_and_import(
        GatherElements(),
        x,
        y,
        dynamic_shapes=dynamic_shapes,
        import_symbolic_shape_expressions=True,
    )
    print(m)


@run
# CHECK-LABEL: test_nonzero
# CHECK:  func.func @main(%[[ARG0:.+]]: !torch.vtensor<[?,3],f32>) -> !torch.vtensor<[?,2],si64> {
# CHECK-DAG:    %[[U0:.+]] = torch.symbolic_int "u0" {min_val = 0, max_val = 9223372036854775806} : !torch.int
# CHECK-DAG:    %[[S0:.+]] = torch.symbolic_int "s0" {min_val = 3, max_val = 10} : !torch.int
# CHECK:        torch.bind_symbolic_shape %[[ARG0]], [%[[S0]]], affine_map<()[s0] -> (s0, 3)> : !torch.vtensor<[?,3],f32>
# CHECK:        %[[NZERO:.+]] = torch.aten.nonzero %[[ARG0]] : !torch.vtensor<[?,3],f32> -> !torch.vtensor<[?,2],si64>
# CHECK:        torch.bind_symbolic_shape %[[NZERO]], [%[[U0]]], affine_map<()[s0] -> (s0, 2)> : !torch.vtensor<[?,2],si64>
# CHECK:        return %[[NZERO]] : !torch.vtensor<[?,2],si64>
def test_nonzero():
    class Nonzero(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nonzero(x)

    # Sample inputs
    x = torch.randn(4, 3)

    # Dynamic dim constraints
    batch = Dim("batch", min=3, max=10)
    dynamic_shapes = {"x": {0: batch}}

    m = fx.export_and_import(
        Nonzero(),
        x,
        dynamic_shapes=dynamic_shapes,
        import_symbolic_shape_expressions=True,
    )
    print(m)
