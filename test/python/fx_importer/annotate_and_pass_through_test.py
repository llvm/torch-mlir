# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

"""Exercise the custom marker op `AnnotateAndPassThrough` during FX import into MLIR."""

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


class LinearWithInputAndOutputAnnotations(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        x = AnnotateAndPassThrough(x, {"my.tag": "input"})
        y = self.fc(x)
        y = AnnotateAndPassThrough(y, {"my.range_lo": -1.0})
        return y


# CHECK-LABEL: test_annotate_and_pass_through_raw
# CHECK:       func.func @main(%arg0: !torch.vtensor<[1,4],f32> {mlir.user.my.tag = "input"})
# CHECK:       torch.aten.linear
# CHECK-SAME:  {mlir.user.my.range_lo = -1.000000e+00 : f64}
# CHECK-NOT:   annotate_and_pass_through
@run
def test_annotate_and_pass_through_raw():
    m = fx.export_and_import(
        LinearWithInputAndOutputAnnotations(),
        torch.randn(1, 4),
        output_type=OutputType.RAW,
    )
    print(m)


# CHECK-LABEL: test_annotate_and_pass_through_linalg
# CHECK:       func.func @main(%arg0: tensor<1x4xf32> {my.tag = "input"})
# CHECK:       linalg.matmul
# CHECK-NOT:   annotate_and_pass_through
@run
def test_annotate_and_pass_through_linalg():
    m = fx.export_and_import(
        LinearWithInputAndOutputAnnotations(),
        torch.randn(1, 4),
        output_type=OutputType.LINALG_ON_TENSORS,
    )
    print(m)
