# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s


import torch

from framework import run_test
from torch_mlir.eager_mode.torch_mlir_dispatch import normalize_args_kwargs


# TODO:Fix me. Tracked with https://github.com/llvm/torch-mlir/issues/789
# CHECK: XFAIL - should_normalize
@run_test(XFAIL=True)
def should_normalize():
    target = torch.ops.aten.max_pool2d_with_indices.default.overloadpacket
    args = (torch.randn((1, 3, 32, 32)),)
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

    new_args, new_kwargs = normalize_args_kwargs(target, args, kwargs)
    for arg, new_arg in zip(args, new_args):
        assert torch.allclose(arg, new_arg)
    for k, v in new_kwargs.items():
        assert v == golden[k]


# TODO:Fix me. Tracked with https://github.com/llvm/torch-mlir/issues/789
# CHECK: XFAIL - shouldnt_normalize1
# CHECK: Couldn't normalize args and kwargs
@run_test(XFAIL=True)
def shouldnt_normalize1():
    target = torch.ops.aten.max_pool2d_with_indices.default.overloadpacket
    args = (torch.randn((1, 3, 32, 32)),)
    kwargs = {"stride": []}
    normalize_args_kwargs(target, args, kwargs)


# TODO:Fix me. Tracked with https://github.com/llvm/torch-mlir/issues/789
# CHECK: XFAIL - shouldnt_normalize2
# CHECK: Couldn't normalize args and kwargs
@run_test(XFAIL=True)
def shouldnt_normalize2():
    target = torch.ops.aten.max_pool2d_with_indices.default.overloadpacket
    args = (torch.randn((1, 3, 32, 32)),)
    kwargs = {"kernel_size": []}
    normalize_args_kwargs(target, args, kwargs)


# TODO:Fix me. Tracked with https://github.com/llvm/torch-mlir/issues/789
# CHECK: XFAIL - shouldnt_normalize3
# CHECK: Couldn't normalize args and kwargs
@run_test(XFAIL=True)
def shouldnt_normalize3():
    target = torch.ops.aten.max_pool2d_with_indices.default.overloadpacket
    args = (torch.randn((1, 3, 32, 32)),)
    kwargs = {"kernel_size": [3, 3], "padding": None}
    normalize_args_kwargs(target, args, kwargs)
