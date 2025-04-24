# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s
# This file contains tests of various op special forms that the fx_importer
# handles.

from typing import Optional

import torch
import torch.export
import torch.nn as nn

from torch_mlir import fx


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# CHECK-LABEL: test_scalar_typed_node
# Getting the shape of a dynamic dimension has the side effect of producing
# a node like:
#   sym_size_int: "Sym(s0)" = torch.ops.aten.sym_size.int(arg0_1, 0)
# This tests the fx_importer code paths around resolving scalar/symbolic
# types for operands and results.
def test_scalar_typed_node():
    class Basic(nn.Module):
        def forward(self, x):
            x = x + 1.0
            return x.shape[0]

    # CHECK: %[[S0:.*]] = torch.symbolic_int "{{[a-z0-9]+}}" {min_val = {{[0-9]+}}, max_val = {{[0-9]+}}} : !torch.int
    # CHECK: torch.bind_symbolic_shape %arg0, [%[[S0]]], affine_map<()[s0] -> (s0, 4)> : !torch.vtensor<[?,4],f32>
    # CHECK: torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,4],f32>, !torch.int -> !torch.int
    m = fx.export_and_import(
        Basic(),
        torch.randn(3, 4),
        dynamic_shapes={"x": {0: torch.export.Dim("b", min=3, max=10)}},
        import_symbolic_shape_expressions=True,
    )
    print(m)
