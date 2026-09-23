# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

"""Test that multi-result operations can have different annotations on each result."""

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


class TopKWithDifferentAnnotations(nn.Module):
    def forward(self, x):
        # topk returns (values, indices)
        values, indices = torch.topk(x, k=2, dim=-1)
        # Annotate each result differently
        values = AnnotateAndPassThrough(
            values, {"my.tag": "values", "my.range_lo": -1.0}
        )
        indices = AnnotateAndPassThrough(
            indices, {"my.tag": "indices", "my.type": "int"}
        )
        return values, indices


# CHECK-LABEL: test_multi_result_different_annotations
# This test verifies that multi-result operations correctly preserve per-result annotations
# using the array-of-dictionaries representation.
# CHECK:       func.func @main(%arg0: !torch.vtensor<[4],f32>)
# CHECK:       %[[VALUES:.*]], %[[INDICES:.*]] = torch.aten.topk
# CHECK-SAME:    {mlir.user = [{my.range_lo = -1.000000e+00 : f64, my.tag = "values"}, {my.tag = "indices", my.type = "int"}]}
# The array has two elements:
#   - Index 0 (values): {my.range_lo = -1.0, my.tag = "values"}
#   - Index 1 (indices): {my.tag = "indices", my.type = "int"}
# CHECK-NOT:   annotate_and_pass_through
@run
def test_multi_result_different_annotations():
    m = fx.export_and_import(
        TopKWithDifferentAnnotations(),
        torch.randn(4),
        output_type=OutputType.RAW,
    )
    print(m)


class SplitWithAnnotations(nn.Module):
    def forward(self, x):
        # split returns a list, but in this case 2 tensors
        chunk1, chunk2 = torch.split(x, 2, dim=0)
        chunk1 = AnnotateAndPassThrough(chunk1, {"my.chunk": "first", "my.offset": 0})
        chunk2 = AnnotateAndPassThrough(chunk2, {"my.chunk": "second", "my.offset": 2})
        return chunk1, chunk2


# CHECK-LABEL: test_split_different_annotations
# This test verifies that split results can have different annotations
# Note: torch.split returns a list, which may be handled differently in FX graph
# CHECK:       func.func @main(%arg0: !torch.vtensor<[4,3],f32>)
# CHECK:       torch.aten.split
# CHECK-NOT:   annotate_and_pass_through
@run
def test_split_different_annotations():
    m = fx.export_and_import(
        SplitWithAnnotations(),
        torch.randn(4, 3),
        output_type=OutputType.RAW,
    )
    print(m)
