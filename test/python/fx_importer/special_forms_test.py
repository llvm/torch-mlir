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
# CHECK-LABEL: test_lift_fresh_copy
def test_lift_fresh_copy():
    #
    class Basic(nn.Module):
        def forward(self, x):
            # CHECK: torch.aten.clone %arg0, %none
            return torch.ops.aten.lift_fresh_copy.default(x)

    m = fx.export_and_import(Basic(), torch.randn(3, 4))
    print(m)
