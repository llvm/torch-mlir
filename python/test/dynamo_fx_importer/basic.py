# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

from typing import List

import torch
import torch.fx
import functorch
import torch._dynamo as dynamo
from torch._dynamo.optimizations.training import aot_autograd

from torch_mlir._dynamo_fx_importer import import_fx_graph_as_func

@functorch.compile.make_boxed_compiler
def my_aot_autograd_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    # TODO: Find a way to get a useful identifier here.
    mlir_module = import_fx_graph_as_func(gm.graph, "MEANINGLESS_NAME")
    print(mlir_module.operation.get_asm(enable_debug_info=True))
    return gm

my_backend = aot_autograd(fw_compiler=my_aot_autograd_backend)

# CHECK:      module attributes {torch.debug_module_name = "MEANINGLESS_NAME"} {
# CHECK-NEXT:   func.func @MEANINGLESS_NAME(%[[ARG0:.*]]: !torch.vtensor<[3,4],f32> loc(unknown)) -> !torch.vtensor<[3,4],f32> {
# CHECK-NEXT:     %[[TANH:.*]] = torch.aten.tanh %[[ARG0]] : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32> loc(#[[LOC:.*]])
# CHECK-NEXT:     return %[[TANH]] : !torch.vtensor<[3,4],f32>
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT: #[[LOC]] = loc("{{.*}}/dynamo_fx_importer/basic.py":{{[0-9]+}}:{{[0-9]+}})
@dynamo.optimize(my_backend)
def basic(x):
    return torch.tanh(x)
basic(torch.randn(3, 4))

# CHECK-LABEL:   func.func @MEANINGLESS_NAME() -> !torch.vtensor<[3,4],f16> {
# CHECK:           %[[INT3:.*]] = torch.constant.int 3
# CHECK:           %[[INT4:.*]] = torch.constant.int 4
# CHECK:           %[[LIST:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT4]] : (!torch.int, !torch.int) -> !torch.list<int>
# CHECK:           %[[INT5:.*]] = torch.constant.int 5
# CHECK:           %[[NONE0:.*]] = torch.constant.none
# CHECK:           %[[DEVICE_CPU:.*]] = torch.constant.device "cpu"
# CHECK:           %[[NONE1:.*]] = torch.constant.none
# CHECK:           %[[RANDN:.*]] = torch.aten.randn %[[LIST]], %[[INT5]], %[[NONE0]], %[[DEVICE_CPU]], %[[NONE1]] : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.none -> !torch.vtensor<[3,4],f16>
# CHECK:           return %[[RANDN]] : !torch.vtensor<[3,4],f16>
@dynamo.optimize(my_backend)
def literals_list_device_int_none_dtype():
    return torch.ops.aten.randn([3, 4], device=torch.device("cpu"), dtype=torch.float16)
literals_list_device_int_none_dtype()

# CHECK-LABEL:   func.func @MEANINGLESS_NAME(
# CHECK-SAME:                                %[[ARG0:.*]]: !torch.vtensor<[3,4],f32> loc(unknown)) -> !torch.vtensor<[3,4],f32> {
# CHECK:           %[[NONE0:.*]] = torch.constant.none
# CHECK:           %[[NONE1:.*]] = torch.constant.none
# CHECK:           %[[NONE2:.*]] = torch.constant.none
# CHECK:           %[[BOOL_FALSE:.*]] = torch.constant.bool false
# CHECK:           %[[NONE3:.*]] = torch.constant.none
# CHECK:           %[[EMPTY_LIKE:.*]] = torch.aten.empty_like %[[ARG0]], %[[NONE0]], %[[NONE1]], %[[NONE2]], %[[BOOL_FALSE]], %[[NONE3]] : !torch.vtensor<[3,4],f32>, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[3,4],f32>
# CHECK:           return %[[EMPTY_LIKE]] : !torch.vtensor<[3,4],f32>
@dynamo.optimize(my_backend)
def literals_bool(x):
    return torch.ops.aten.empty_like(x, pin_memory=False)
literals_bool(torch.randn(3, 4))

# CHECK-LABEL:   func.func @MEANINGLESS_NAME(
# CHECK-SAME:                                %[[ARG0:.*]]: !torch.vtensor<[3,4],f32> loc(unknown)) -> !torch.vtensor<[3,4],f32> {
# CHECK:           %[[FLOAT0:.*]] = torch.constant.float 0.000000e+00
# CHECK:           %[[FLOAT1:.*]] = torch.constant.float 1.000000e+00
# CHECK:           %[[NONE:.*]] = torch.constant.none
# CHECK:           %[[UNIFORM:.*]] = torch.aten.uniform %[[ARG0]], %[[FLOAT0]], %[[FLOAT1]], %[[NONE]] : !torch.vtensor<[3,4],f32>, !torch.float, !torch.float, !torch.none -> !torch.vtensor<[3,4],f32>
# CHECK:           return %[[UNIFORM]] : !torch.vtensor<[3,4],f32>
@dynamo.optimize(my_backend)
def literals_float(x):
    return torch.ops.aten.uniform(x, 0.0, 1.0)
literals_float(torch.randn(3, 4))

# CHECK-LABEL:   func.func @MEANINGLESS_NAME(
# CHECK-SAME:                                %[[ARG0:.*]]: !torch.vtensor<[3,4],f32> loc(unknown)) -> !torch.vtensor<[3,4],f32> {
# CHECK:           %[[STR_TANH:.*]] = torch.constant.str "tanh"
# CHECK:           %[[GELU:.*]] = torch.aten.gelu %[[ARG0]], %[[STR_TANH]] : !torch.vtensor<[3,4],f32>, !torch.str -> !torch.vtensor<[3,4],f32>
# CHECK:           return %[[GELU]] : !torch.vtensor<[3,4],f32>
@dynamo.optimize(my_backend)
def literals_str(x):
    return torch.ops.aten.gelu(x, approximate="tanh")
literals_str(torch.randn(3, 4))

# CHECK-LABEL:   func.func @MEANINGLESS_NAME(
# CHECK-SAME:                                %[[ARG0:.*]]: !torch.vtensor<[3,4],f32> loc(unknown)) -> !torch.vtensor<[3,4],f32> {
# CHECK:           %[[NONE:.*]] = torch.constant.none
# CHECK:           %[[CLONE:.*]] = torch.aten.clone %[[ARG0]], %[[NONE]] : !torch.vtensor<[3,4],f32>, !torch.none -> !torch.vtensor<[3,4],f32>
# CHECK:           return %[[CLONE]] : !torch.vtensor<[3,4],f32>
@dynamo.optimize(my_backend)
def default_overload(x):
    return torch.ops.aten.clone.default(x)
default_overload(torch.randn(3, 4))

# CHECK-LABEL:   func.func @MEANINGLESS_NAME(
# CHECK-SAME:                                %[[ARG0:.*]]: !torch.vtensor<[3,4],f32> loc(unknown)) -> (!torch.vtensor<[4],f32>, !torch.vtensor<[4],si64>) {
# CHECK:           %[[INT0:.*]] = torch.constant.int 0 loc(#loc1)
# CHECK:           %[[FALSE:.*]] = torch.constant.bool false loc(#loc1)
# CHECK:           %[[VALUES:.*]], %[[INDICES:.*]] = torch.aten.max.dim %[[ARG0]], %[[INT0]], %[[FALSE]] : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool -> !torch.vtensor<[4],f32>, !torch.vtensor<[4],si64> loc(#loc1)
# CHECK:           return %[[VALUES]], %[[INDICES]] : !torch.vtensor<[4],f32>, !torch.vtensor<[4],si64> loc(#loc)
@dynamo.optimize(my_backend)
def multiple_results(x):
    return torch.ops.aten.max.dim(x, 0)
multiple_results(torch.randn(3, 4))

@dynamo.optimize(my_backend)
def clone(x):
    return x.clone()
clone(torch.randn(()))
