# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

import typing

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

# CHECK-LABEL:   func @__torch__.optional_return(
# CHECK-SAME:                            %[[ARG:.*]]: i64) -> !torch.optional<i64> {
# CHECK:           %[[RET:.*]] = torch.derefine %[[ARG]] : i64 to !torch.optional<i64>
# CHECK:           return %[[RET]] : !torch.optional<i64>
@mb.import_function
@torch.jit.script
def optional_return(i: int) -> typing.Optional[int]:
    return i

# CHECK-LABEL:   func @__torch__.optional_arg(
# CHECK-SAME:                                      %[[ARG:.*]]: !torch.optional<i64>) -> !torch.none {
@mb.import_function
@torch.jit.script
def optional_arg(i: typing.Optional[int]) -> None:
    return

# CHECK-LABEL:   func @__torch__.calls_optional_arg(
# CHECK-SAME:                                       %[[ARG:.*]]: i64) -> !torch.none {
# CHECK:           %[[CALLEE:.*]] = constant @__torch__.optional_arg : (!torch.optional<i64>) -> !torch.none
# CHECK:           %[[DEREFINED:.*]] = torch.derefine %[[ARG]] : i64 to !torch.optional<i64>
# CHECK:           %{{.*}} = call_indirect %[[CALLEE]](%[[DEREFINED]]) : (!torch.optional<i64>) -> !torch.none
@mb.import_function
@torch.jit.script
def calls_optional_arg(i: int):
    optional_arg(i)


mb.module.operation.print()
print()
