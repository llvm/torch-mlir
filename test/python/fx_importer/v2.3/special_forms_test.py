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

################################################################################
# Custom ops to test various things that are hard to reach.
################################################################################
LIBRARY = torch.library.Library("torch_mlir_test", "DEF")

LIBRARY.define("multi_return(Tensor x) -> (Tensor, Tensor, Tensor)")


def multi_return_meta(x):
    return None, torch.empty_like(x), torch.empty_like(x)


LIBRARY.impl("multi_return", multi_return_meta, "Meta")


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# CHECK-LABEL: test_lift_fresh_copy
def test_lift_fresh_copy():
    class Basic(nn.Module):
        def forward(self, x):
            # CHECK: torch.aten.clone %arg0, %none
            return torch.ops.aten.lift_fresh_copy.default(x)

    m = fx.export_and_import(Basic(), torch.randn(3, 4))
    print(m)


@run
# CHECK-LABEL: test_multi_return
def test_multi_return():
    class Basic(nn.Module):
        def forward(self, x):
            # Note that optional return tensors that are statically traced to
            # None show up as a !torch.none type. This happens in the case of
            # certain convolution backwards ops (possibly among others).
            # The FxImporter does not perform special tracking of static None
            # values, instead just materializing a torch.constant.none when
            # needed. This is an implementation detail: it would be valid to
            # use the RES:0 result instead of this materialization below.
            # In practice, this doesn't arise in nature and is a by-product
            # of tracing.
            # CHECK: %[[RES:.*]]:3 = torch.operator "torch.torch_mlir_test.multi_return"(%arg0) :
            # CHECK-SAME: (!torch.vtensor<[3,4],f32>)
            # CHECK-SAME: -> (!torch.none, !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32>)
            # CHECK: %[[NONE:.*]] = torch.constant.none
            # CHECK: return %[[NONE]], %[[RES]]#1, %[[RES]]#2
            return torch.ops.torch_mlir_test.multi_return(x)

    m = fx.export_and_import(Basic(), torch.randn(3, 4))
    print(m)
