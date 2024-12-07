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


LIBRARY = torch.library.Library("torch_mlir_test", "DEF")

LIBRARY.define("inplace_modify(Tensor(a!) x) -> ()")
LIBRARY.define("inplace_modify_calc(Tensor(a!) x) -> (Tensor)")


def inplace_modify_calc_meta(x):
    return torch.empty_like(x)


LIBRARY.impl("inplace_modify_calc", inplace_modify_calc_meta, "Meta")


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


# CHECK-LABEL: test_auto_functionalized_hop
@run
def test_auto_functionalized_hop():
    class Basic(nn.Module):
        def forward(self, x):
            torch.ops.torch_mlir_test.inplace_modify(x)
            return x * x

    m = fx.export_and_import(
        Basic(),
        torch.randn(3, 4),
        experimental_support_mutation=True,
        # TODO: ExportedProgram.run_decompositions() seems to have trouble
        # with mode selection and Python higher order op implementations.
        # Isolate and report upstream.
        # Raises:
        #   File "/home/stella/v/Dev/lib/python3.11/site-packages/torch/_ops.py", line 323, in dispatch
        # assert (
        # AssertionError: Current active mode <torch._subclasses.functional_tensor.FunctionalTensorMode object at 0x7a1106504fd0> not registered
        decomposition_table=[],
    )
    # The Torch 2.6 expects the IR to be same as the below one, while the torch versions < 2.6 does not, hence this check is kept as a "COM".
    # COM: torch.operator "torch.torch_mlir_test.inplace_modify"({{.*}}) : (!torch.vtensor<[3,4],f32>) -> ()
    # CHECK: torch.aten.mul.Tensor %{{.*}}, %{{.*}}
    print(m)
    m.operation.verify()


# CHECK-LABEL: test_auto_functionalized_one_ret
@run
def test_auto_functionalized_one_ret():
    class Basic(nn.Module):
        def forward(self, x):
            y = torch.ops.torch_mlir_test.inplace_modify_calc(x)
            return x * y

    m = fx.export_and_import(
        Basic(),
        torch.randn(3, 4),
        experimental_support_mutation=True,
        # TODO: ExportedProgram.run_decompositions() seems to have trouble
        # with mode selection and Python higher order op implementations.
        # Isolate and report upstream.
        # Raises:
        #   File "/home/stella/v/Dev/lib/python3.11/site-packages/torch/_ops.py", line 323, in dispatch
        # assert (
        # AssertionError: Current active mode <torch._subclasses.functional_tensor.FunctionalTensorMode object at 0x7a1106504fd0> not registered
        decomposition_table=[],
    )
    # The Torch 2.6 expects the IR to be same as the below one, while the torch versions < 2.6 does not, hence this check is kept as a "COM".
    # COM: %[[TIED:.*]] = torch.operator "torch.torch_mlir_test.inplace_modify_calc"(%arg0) : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32>
    # CHECK: torch.aten.mul.Tensor %{{.*}}, %{{.*}}
    print(m)
    m.operation.verify()
