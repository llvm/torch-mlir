# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

"""Test that user annotations survive through the Torch-to-Torch decomposition pipeline."""

import torch
import torch.nn as nn

from torch_mlir import fx
from torch_mlir.fx import OutputType
from torch_mlir.extras.annotate import AnnotateAndPassThrough


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


class LinearWithOutputAnnotation(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        y = self.fc(x)
        y = AnnotateAndPassThrough(y, {"my.range_lo": -1.0, "my.range_hi": 1.0})
        return y


# CHECK-LABEL: test_annotation_survives_decomposition_torch
# This test verifies that annotations survive through the Torch-to-Torch pipeline
# which includes decomposition of aten.linear into transpose/matmul/add.
# With the new per-result forwarding, only the final operation gets the attributes.
# CHECK:       func.func @main(%arg0: !torch.vtensor<[1,4],f32>)
# CHECK:       torch.aten.t
# CHECK-NOT:   mlir.user
# CHECK:       torch.aten.matmul
# CHECK-NOT:   mlir.user
# CHECK:       torch.aten.add.Tensor{{.*}}{mlir.user.my.range_hi = 1.000000e+00 : f64, mlir.user.my.range_lo = -1.000000e+00 : f64}
# CHECK-NOT:   annotate_and_pass_through
@run
def test_annotation_survives_decomposition_torch():
    """Test that annotations survive through decomposition in TORCH output."""
    m = fx.export_and_import(
        LinearWithOutputAnnotation(),
        torch.randn(1, 4),
        output_type=OutputType.TORCH,
    )
    print(m)


# CHECK-LABEL: test_annotation_survives_decomposition_linalg
# This test verifies that annotations survive through the full pipeline
# including both Torch-to-Torch decomposition and Torch-to-Linalg conversion.
# The annotations are forwarded through decomposition and backend conversion.
# CHECK:       func.func @main(%arg0: tensor<1x4xf32>)
# CHECK:       {my.range_hi = 1.000000e+00 : f64, my.range_lo = -1.000000e+00 : f64}
# CHECK-NOT:   annotate_and_pass_through
@run
def test_annotation_survives_decomposition_linalg():
    """Test that annotations survive through the full pipeline to Linalg."""
    m = fx.export_and_import(
        LinearWithOutputAnnotation(),
        torch.randn(1, 4),
        output_type=OutputType.LINALG_ON_TENSORS,
    )
    print(m)
