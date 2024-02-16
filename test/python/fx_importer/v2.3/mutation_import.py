# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

from typing import Optional

import torch
import torch.export
import torch.nn as nn

from torch_mlir import fx

from torch_mlir.ir import (
    Operation,
)


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# Tests that constants and parameters work generally with the mutation path.
# This doesn't do mutation but ensures that the basics remain functional.
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
def test_import_frozen_exported_program():
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

    m = fx.export_and_import(
        Basic(), torch.randn(3, 4), experimental_support_mutation=True
    )
    print(m)
    m.operation.verify()


@run
# CHECK-LABEL: test_user_input_mutate
# CHECK:     func.func @main(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.tensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32>
# CHECK-DAG: %[[arg1_copy:.+]] = torch.copy.to_vtensor %arg1 : !torch.vtensor<[3,4],f32>
# CHECK-DAG: %[[arg1_mul:.+]] = torch.aten.mul.Tensor %[[arg1_copy]], %arg0
# CHECK-DAG: torch.overwrite.tensor.contents %[[arg1_mul]] overwrites %arg1
# CHECK-DAG: %[[arg0_mul:.+]] = torch.aten.mul.Tensor %arg0, %[[arg1_mul]]
# CHECK:     return %[[arg0_mul]]
def test_user_input_mutate():
    class Basic(nn.Module):
        def forward(self, x, y):
            y.mul_(x)
            return x * y

    m = fx.export_and_import(
        Basic(),
        torch.randn(3, 4),
        torch.randn(3, 4),
        experimental_support_mutation=True,
    )
    print(m)
    m.operation.verify()


@run
# CHECK-LABEL: test_frozen_buffer
# CHECK: %[[buffer_literal:.+]] = torch.vtensor.literal
# CHECK: %[[mul:.+]] = torch.aten.mul.Tensor %arg0, %0
# CHECK: return %[[mul]]
def test_frozen_buffer():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buffer", torch.randn(3, 4))

        def forward(self, x):
            return x * self.buffer

    m = fx.export_and_import(
        Basic(), torch.randn(3, 4), experimental_support_mutation=True
    )
    print(m)
    m.operation.verify()


class ExternalBufferHooks(fx.FxImporterHooks):
    def prepare_module(self, module_op: Operation):
        module_op.context.allow_unregistered_dialects = True

    def resolve_input(self, gni, value, info):
        return Operation.create(
            "my_dialect.import_buffer", results=[info.ir_type]
        ).result


@run
# CHECK-LABEL: test_mutable_buffer
# CHECK: %[[buffer:.+]] = "my_dialect.import_buffer"() : () -> !torch.tensor<[3,4],f32>
# CHECK: %[[mul:.+]] = torch.aten.mul.Tensor %[[buffer]], %arg0
# CHECK: torch.overwrite.tensor.contents %[[mul]] overwrites %[[buffer]]
# CHECK: return %arg0
def test_mutable_buffer():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buffer", torch.randn(3, 4))

        def forward(self, x):
            self.buffer.mul_(x)
            return x

    m = fx.export_and_import(
        Basic(),
        torch.randn(3, 4),
        experimental_support_mutation=True,
        hooks=ExternalBufferHooks(),
    )
    print(m)
    m.operation.verify()


@run
# CHECK-LABEL: test_mutable_buffer_not_supported_from_literal
# CHECK: ERROR: Cannot import {{.*}} as a literal because it is mutable
def test_mutable_buffer_not_supported_from_literal():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buffer", torch.randn(3, 4))

        def forward(self, x):
            self.buffer.mul_(x)
            return x

    try:
        m = fx.export_and_import(
            Basic(),
            torch.randn(3, 4),
            experimental_support_mutation=True,
        )
    except ValueError as e:
        print("ERROR:", e)
