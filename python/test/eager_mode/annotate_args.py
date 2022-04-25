# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s


import torch

from framework import run_test
from torch_mlir.eager_mode.torch_mlir_dispatch import (
    annotate_args_kwargs,
    normalize_args_kwargs,
    build_script_function,
)


# CHECK: Torch Tensor (shape=(1, 3, 32, 32), dtype=torch.float32)
# CHECK: Torch Tensor (shape=(1, 3, 32, 32), dtype=torch.float32)
# CHECK: Torch Tensor (shape=(1, 3, 32, 32), dtype=torch.float32)
# -----
# CHECK: PASS - simple
# TODO:Fix me. Tracked with https://github.com/llvm/torch-mlir/issues/789
@run_test(XFAIL=True)
def simple():
    target = torch.ops.aten.addmm.default
    A = torch.randn(1, 3, 32, 32)
    B = torch.randn(1, 3, 32, 32)
    C = torch.randn(1, 3, 32, 32)
    args = (A, B, C)
    kwargs = dict(beta=1, alpha=1)

    new_args, new_kwargs = normalize_args_kwargs(target.overloadpacket, args, kwargs)
    script_fun = build_script_function(target._schema, new_args, new_kwargs)
    annotations, *_ = annotate_args_kwargs(script_fun, new_args, new_kwargs)
    for annot in annotations:
        print(annot)


# CHECK: Torch Tensor (shape=(-1, 3, 32, 32), dtype=torch.float32)
# CHECK: Torch Tensor (shape=(-1, 3, 32, 32), dtype=torch.float32)
# CHECK: Torch Tensor (shape=(-1, 3, 32, 32), dtype=torch.float32)
# -----
# CHECK: PASS - handle_zero_dim
# TODO:Fix me. Tracked with https://github.com/llvm/torch-mlir/issues/789
@run_test(XFAIL=True)
def handle_zero_dim():
    target = torch.ops.aten.addmm.default
    A = torch.randn(0, 3, 32, 32)
    B = torch.randn(0, 3, 32, 32)
    C = torch.randn(0, 3, 32, 32)
    args = (A, B, C)
    kwargs = dict(beta=1, alpha=1)

    new_args, new_kwargs = normalize_args_kwargs(target.overloadpacket, args, kwargs)
    script_fun = build_script_function(target._schema, new_args, new_kwargs)
    annotations, *_ = annotate_args_kwargs(script_fun, new_args, new_kwargs)
    for annot in annotations:
        print(annot)


# CHECK: Torch Tensor (shape=(2, 5, 2, 3), dtype=torch.float32)
# CHECK: Torch Tensor (shape=(5,), dtype=torch.float32)
# CHECK: Torch Tensor (shape=(5,), dtype=torch.float32)
# CHECK: Torch Tensor (shape=(5,), dtype=torch.float32)
# CHECK: Torch Tensor (shape=(5,), dtype=torch.float32)
# CHECK: Torch Tensor (shape=(2, 5, 2, 3), dtype=torch.float32)
# CHECK: Torch Tensor (shape=(5,), dtype=torch.float32)
# CHECK: Torch Tensor (shape=(5,), dtype=torch.float32)
# -----
# CHECK: PASS - correctly_order_kwargs
# TODO:Fix me. Tracked with https://github.com/llvm/torch-mlir/issues/789
@run_test(XFAIL=True)
def correctly_order_kwargs():
    target = torch.ops.aten.native_batch_norm.out

    input = torch.randn(2, 5, 2, 3)
    weight = torch.randn(5)
    bias = torch.randn(5)
    running_mean = torch.randn(5)
    running_var = torch.randn(5)
    args = (input, weight, bias, running_mean, running_var)

    out = torch.empty_like(input)
    save_mean = torch.empty_like(running_mean)
    save_invstd = torch.empty_like(running_var)

    kwargs = dict(
        training=False,
        momentum=0.1,
        eps=0.0001,
        out=out,
        save_mean=save_mean,
        save_invstd=save_invstd,
    )

    new_args, new_kwargs = normalize_args_kwargs(target.overloadpacket, args, kwargs)
    script_fun = build_script_function(target._schema, new_args, new_kwargs)
    annotations, *_ = annotate_args_kwargs(script_fun, new_args, new_kwargs)
    for annot in annotations:
        print(annot)
