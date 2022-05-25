# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s


import torch

from framework import run_test
from torch_mlir.eager_mode.torch_mlir_dispatch import normalize_args_kwargs


# CHECK: PASS - should_normalize
@run_test
def should_normalize():
    target = torch.ops.aten.max_pool2d_with_indices.default
    input = torch.randn((1, 3, 32, 32))
    kwargs = {"kernel_size": [3, 3]}
    golden = {
        "kernel_size": [3, 3],
        # This is due to the schema for max_pool2d_with_indices defining
        # the stride arg as int[2] stride=[].
        "stride": [],
        "padding": [0, 0],
        "dilation": [1, 1],
        "ceil_mode": False,
    }

    new_kwargs = normalize_args_kwargs(target, (input,), kwargs)
    assert torch.allclose(new_kwargs["input"], input)
    for k, v in new_kwargs.items():
        if k == "input": continue
        assert v == golden[k]


# CHECK: FAIL - shouldnt_normalize1
# CHECK: Errors: missing a required argument: 'kernel_size'
@run_test
def shouldnt_normalize1():
    target = torch.ops.aten.max_pool2d_with_indices.default
    args = (torch.randn((1, 3, 32, 32)),)
    kwargs = {"stride": []}
    normalize_args_kwargs(target, args, kwargs)


# This next two tests are XPASS because of https://github.com/pytorch/pytorch/issues/75342
# I.e., they should fail but in fact they pass because of the upstream bug.
# The reason for the bug is a fast path branch in operator_schemas.normalize_function
# that doesn't do rigorous type checking, and hence lets type mistmatches slip through.
# TODO(max): change these to FAIL when the upstream bug is fixed.

# CHECK: XPASS - shouldnt_normalize2
@run_test(XPASS=True)
def shouldnt_normalize2():
    target = torch.ops.aten.max_pool2d_with_indices.default
    args = (torch.randn((1, 3, 32, 32)),)
    kwargs = {"kernel_size": []}
    normalize_args_kwargs(target, args, kwargs)


# CHECK: XPASS - shouldnt_normalize3
@run_test(XPASS=True)
def shouldnt_normalize3():
    target = torch.ops.aten.max_pool2d_with_indices.default
    args = (torch.randn((1, 3, 32, 32)),)
    kwargs = {"kernel_size": [3, 3], "padding": None}
    normalize_args_kwargs(target, args, kwargs)
