# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch
import torch.nn as nn

from torch_mlir import fx


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# CHECK-LABEL: test_unbind_int_op
# CHECK:     func.func @test_unbind_int(%[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[3,4],f32>) -> (!torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>)
# CHECK-DAG:   %[[unbind:.+]] = torch.aten.unbind.int %[[ARG0]]
# CHECK-DAG:   %[[idx0:.+]] = torch.constant.int 0
# CHECK-DAG:   %[[result0:.+]] = torch.aten.__getitem__.t %[[unbind]], %[[idx0]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[3],f32>
# CHECK:       return %[[result0]],
def test_unbind_int_op():
    class UnbindIntModule(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.unbind(x, 1)

    m = fx.export_and_import(
        UnbindIntModule(),
        torch.randn(3, 4),
        func_name="test_unbind_int",
        decomposition_table=False,
    )
    print(m)
