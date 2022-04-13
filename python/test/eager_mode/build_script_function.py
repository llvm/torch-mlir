# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s


import torch

from framework import run_test
from torch_mlir.eager_mode.ir_building import build_ts_script_function


# CHECK: graph(%[[A1:.*]] : Tensor,
# CHECK:  %[[A2:.*]] : Tensor,
# CHECK:  %[[A3:.*]] : Tensor):
# CHECK:  %[[A4:.*]] : int = prim::Constant[value=1]()
# CHECK:  %[[A5:.*]] : int = prim::Constant[value=1]()
# CHECK:  %[[A0:.*]] : Tensor = aten::addmm(%[[A1]], %[[A2]], %[[A3]], %[[A4]], %[[A5]])
# CHECK:  return (%[[A0]])
# -----
# CHECK: PASS - simple
@run_test
def simple():
    target = torch.ops.aten.addmm.default
    kwargs = dict(
        input=torch.randn(1, 3, 32, 32),
        mat1=torch.randn(1, 3, 32, 32),
        mat2=torch.randn(1, 3, 32, 32),
        beta=1,
        alpha=1,
    )

    script_fun = build_ts_script_function(target._schema, kwargs)
    print(script_fun.graph)


# CHECK: graph(%[[B1:.*]] : Tensor,
# CHECK:  %[[B2:.*]] : Tensor,
# CHECK:  %[[B3:.*]] : Tensor):
# CHECK:  %[[B4:.*]] : int[] = prim::Constant[value=[1, 1]]()
# CHECK:  %[[B5:.*]] : int[] = prim::Constant[value=[0, 0]]()
# CHECK:  %[[B6:.*]] : int[] = prim::Constant[value=[1, 1]]()
# CHECK:  %[[B7:.*]] : bool = prim::Constant[value=0]()
# CHECK:  %[[B8:.*]] : int[] = prim::Constant[value=[0, 0]]()
# CHECK:  %[[B9:.*]] : int = prim::Constant[value=1]()
# CHECK:  %[[B0:.*]] : Tensor = aten::convolution(%[[B1]], %[[B2]], %[[B3]], %[[B4]], %[[B5]], %[[B6]], %[[B7]], %[[B8]], %[[B9]])
# CHECK:  return (%[[B0]])
# -----
# CHECK: PASS - handle_optional_tensor_input
@run_test
def handle_optional_tensor_input():
    target = torch.ops.aten.convolution.default
    kwargs = dict(
        input=torch.randn(1, 3, 32, 32),
        weight=torch.randn(3, 3, 3, 3),
        bias=torch.randn(3),
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        transposed=False,
        output_padding=[0, 0],
        groups=1,
    )
    script_fun = build_ts_script_function(target._schema, kwargs)
    print(script_fun.graph)


# CHECK: FAIL - fail_not_enough_args
# CHECK: Errors:  'groups'
@run_test
def fail_not_enough_args():
    target = torch.ops.aten.convolution.default
    kwargs = dict(
        input=torch.randn(1, 3, 32, 32),
        weight=torch.randn(3, 3, 3, 3),
        bias=torch.randn(3),
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        transposed=False,
        output_padding=[0, 0],
        # Missing groups=1,
    )
    build_ts_script_function(target._schema, kwargs)


# CHECK: graph(%input : Tensor,
# CHECK:  %weight : Tensor,
# CHECK:  %bias : Tensor):
# CHECK:  %4 : int[] = prim::Constant[value=[1, 1]]()
# CHECK:  %5 : int[] = prim::Constant[value=[0, 0]]()
# CHECK:  %6 : int[] = prim::Constant[value=[1, 1]]()
# CHECK:  %7 : bool = prim::Constant[value=0]()
# CHECK:  %8 : int[] = prim::Constant[value=[0, 0]]()
# CHECK:  %9 : int = prim::Constant[value=1]()
# CHECK:  %0 : Tensor = aten::convolution(%input, %weight, %bias, %4, %5, %6, %7, %8, %9)
# CHECK:  return (%0)
# -----
# CHECK: PASS - simple_kwargs
@run_test
def simple_kwargs():
    target = torch.ops.aten.convolution.default
    script_fun1 = build_ts_script_function(
        target._schema,
        dict(
            input=torch.randn(1, 3, 32, 32),
            weight=torch.randn(3, 3, 3, 3),
            bias=torch.randn(3),
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=1,
        ),
    )

    print(script_fun1.graph)


# CHECK: graph(%[[C2:.*]] : Tensor):
# CHECK:   %[[C3:.*]] : int[] = prim::Constant[value=[3, 3]]()
# CHECK:   %[[C4:.*]] : NoneType = prim::Constant()
# CHECK:   %[[C5:.*]] : int[] = prim::Constant[value=[0, 0]]()
# CHECK:   %[[C6:.*]] : int[] = prim::Constant[value=[1, 1]]()
# CHECK:   %[[C7:.*]] : bool = prim::Constant[value=0]()
# CHECK:   %[[C0:.*]] : Tensor, %[[C1:.*]] : Tensor = aten::max_pool2d_with_indices(%[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[C6]], %[[C7]])
# CHECK:   return (%[[C0]], %[[C1]])
# -----
# CHECK: PASS - handle_empty_lists
@run_test
def handle_empty_lists():
    target = torch.ops.aten.max_pool2d_with_indices.default
    # print(target._schema)
    input = torch.randn((1, 3, 32, 32))
    kwargs = dict(
        input=input,
        kernel_size=[3, 3],
        stride=[],
        padding=[0, 0],
        dilation=[1, 1],
        ceil_mode=False,
    )
    script_fun = build_ts_script_function(target._schema, kwargs)
    print(script_fun.graph)


# CHECK: graph(%[[D2:.*]] : Tensor):
# CHECK:   %[[D3:.*]] : int[] = prim::Constant[value=[3, 3]]()
# CHECK:   %[[D4:.*]] : NoneType = prim::Constant()
# CHECK:   %[[D5:.*]] : int[] = prim::Constant[value=[0, 0]]()
# CHECK:   %[[D6:.*]] : int[] = prim::Constant[value=[1, 1]]()
# CHECK:   %[[D7:.*]] : bool = prim::Constant[value=0]()
# CHECK:   %[[D0:.*]] : Tensor, %[[D1:.*]] : Tensor = aten::max_pool2d_with_indices(%[[D2]], %[[D3]], %[[D4]], %[[D5]], %[[D6]], %[[D7]])
# CHECK:   return (%[[D0]], %[[D1]])
# -----
# CHECK: PASS - handle_nones
@run_test
def handle_nones():
    target = torch.ops.aten.max_pool2d_with_indices.default
    # print(target._schema)
    kwargs = dict(
        input=torch.randn((1, 3, 32, 32)),
        kernel_size=[3, 3],
        stride=None,
        padding=[0, 0],
        dilation=[1, 1],
        ceil_mode=False,
    )
    script_fun = build_ts_script_function(target._schema, kwargs)
    print(script_fun.graph)


# CHECK: graph(%[[E1:.*]] : Tensor,
# CHECK:  %[[E2:.*]] : Tensor,
# CHECK:  %[[E3:.*]] : Tensor):
# CHECK:  %[[E4:.*]] : int[] = prim::Constant[value=[1, 1]]()
# CHECK:  %[[E5:.*]] : int[] = prim::Constant[value=[0, 0]]()
# CHECK:  %[[E6:.*]] : int[] = prim::Constant[value=[1, 1]]()
# CHECK:  %[[E7:.*]] : bool = prim::Constant[value=0]()
# CHECK:  %[[E8:.*]] : int[] = prim::Constant[value=[0, 0]]()
# CHECK:  %[[E9:.*]] : int = prim::Constant[value=1]()
# CHECK:  %[[E0:.*]] : Tensor = aten::convolution(%[[E1]], %[[E2]], %[[E3]], %[[E4]], %[[E5]], %[[E6]], %[[E7]], %[[E8]], %[[E9]])
# CHECK:  return (%[[E0]])
# -----
# CHECK: PASS - handle_optional_tensors
@run_test
def handle_optional_tensors():
    target = torch.ops.aten.convolution.default
    kwargs = dict(
        input=torch.randn(1, 3, 32, 32),
        weight=torch.randn(3, 3, 3, 3),
        bias=torch.randn(3),
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        transposed=False,
        output_padding=[0, 0],
        groups=1,
    )
    script_fun = build_ts_script_function(target._schema, kwargs)
    print(script_fun.graph)


# CHECK: graph(%[[F1:.*]] : Tensor):
# CHECK:  %[[F2:.*]] : NoneType = prim::Constant()
# CHECK:  %[[F3:.*]] : NoneType = prim::Constant()
# CHECK:  %[[F4:.*]] : NoneType = prim::Constant()
# CHECK:  %[[F5:.*]] : NoneType = prim::Constant()
# CHECK:  %[[F6:.*]] : NoneType = prim::Constant()
# CHECK:  %[[F0:.*]] : Tensor = aten::ones_like(%[[F1]], %[[F2]], %[[F3]], %[[F4]], %[[F5]], %[[F6]])
# CHECK:  return (%[[F0]])
# -----
# CHECK: PASS - handle_ones_like
@run_test
def handle_ones_like():
    target = torch.ops.aten.ones_like.default
    kwargs = dict(
        input=torch.randn(1, 3, 32, 32),
        dtype=None,
        layout=None,
        device=None,
        pin_memory=None,
        memory_format=None,
    )
    script_fun = build_ts_script_function(target._schema, kwargs)
    print(script_fun.graph)


# CHECK: graph(%[[G3:.*]] : Tensor,
# CHECK:  %[[G4:.*]] : Tensor,
# CHECK:  %[[G5:.*]] : Tensor):
# CHECK:  %[[G6:.*]] : NoneType = prim::Constant()
# CHECK:  %[[G7:.*]] : NoneType = prim::Constant()
# CHECK:  %[[G8:.*]] : bool = prim::Constant[value=0]()
# CHECK:  %[[G9:.*]] : float = prim::Constant[value=1.]()
# CHECK:  %[[G10:.*]] : float = prim::Constant[value=1.]()
# CHECK:  %[[G0:.*]] : Tensor, %[[G1:.*]] : Tensor, %[[G2:.*]] : Tensor = aten::native_batch_norm(%[[G3]], %[[G4]], %[[G5]], %[[G6]], %[[G7]], %[[G8]], %[[G9]], %[[G10]])
# CHECK:  return (%[[G0]], %[[G1]], %[[G2]])
# -----
# CHECK: PASS - handle_multiple_outputs
@run_test
def handle_multiple_outputs():
    target = torch.ops.aten.native_batch_norm.default
    kwargs = dict(
        input=torch.randn(1, 3, 32, 32),
        weight=torch.randn(1, 3, 32, 32),
        bias=torch.randn(1, 3, 32, 32),
        running_mean=None,
        running_var=None,
        training=False,
        momentum=1.0,
        eps=1.0
    )

    script_fun = build_ts_script_function(target._schema, kwargs)
    print(script_fun.graph)


# CHECK: f
# CHECK: PASS - check_legal_name
@run_test
def check_legal_name():
    target = torch.ops.aten.native_batch_norm.default
    kwargs = dict(
        input=torch.randn(1, 3, 32, 32),
        weight=torch.randn(1, 3, 32, 32),
        bias=torch.randn(1, 3, 32, 32),
        running_mean=None,
        running_var=None,
        training=False,
        momentum=1.0,
        eps=1.0
    )

    script_fun = build_ts_script_function(target._schema, kwargs)
    print(script_fun.name)


# CHECK: graph(%[[H3:.*]] : Tensor,
# CHECK:  %[[H4:.*]] : Tensor,
# CHECK:  %[[H5:.*]] : Tensor,
# CHECK:  %[[H6:.*]] : Tensor,
# CHECK:  %[[H7:.*]] : Tensor,
# CHECK:  %out : Tensor,
# CHECK:  %save_mean : Tensor,
# CHECK:  %save_invstd : Tensor):
# CHECK:  %[[H8:.*]] : bool = prim::Constant[value=0]()
# CHECK:  %[[H9:.*]] : float = prim::Constant[value=0.10000000000000001]()
# CHECK:  %[[H10:.*]] : float = prim::Constant[value=0.0001]()
# CHECK:  %[[H0:.*]] : Tensor, %[[H1:.*]] : Tensor, %[[H2:.*]] : Tensor = aten::native_batch_norm(%[[H3]], %[[H4]], %[[H5]], %[[H6]], %[[H7]], %[[H8]], %[[H9]], %[[H10]], %out, %save_mean, %save_invstd)
# CHECK:  return (%[[H0]], %[[H1]], %[[H2]])
# -----
# CHECK: PASS - correctly_order_kwargs
@run_test
def correctly_order_kwargs():
    target = torch.ops.aten.native_batch_norm.out

    input = torch.randn(2, 5, 2, 3)
    running_mean = torch.randn(5)
    running_var = torch.randn(5)

    kwargs = dict(
        input=torch.randn(2, 5, 2, 3),
        weight=torch.randn(5),
        bias=torch.randn(5),
        running_mean=running_mean,
        running_var=running_var,
        training=False,
        momentum=0.1,
        eps=0.0001,
        out=torch.empty_like(input),
        save_mean=torch.empty_like(running_mean),
        save_invstd=torch.empty_like(running_var),
    )

    script_fun = build_ts_script_function(target._schema, kwargs)
    print(script_fun.graph)
