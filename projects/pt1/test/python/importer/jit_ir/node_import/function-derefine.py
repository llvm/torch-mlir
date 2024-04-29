# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import torch
from torch_mlir.jit_ir_importer import ModuleBuilder

import typing

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()


# CHECK-LABEL:   func.func @__torch__.optional_return(
# CHECK-SAME:                            %[[ARG:.*]]: !torch.int) -> !torch.optional<int> {
# CHECK:           %[[RET:.*]] = torch.derefine %[[ARG]] : !torch.int to !torch.optional<int>
# CHECK:           return %[[RET]] : !torch.optional<int>
@mb.import_function
@torch.jit.script
def optional_return(i: int) -> typing.Optional[int]:
    return i


# CHECK-LABEL:   func.func @__torch__.optional_arg(
# CHECK-SAME:                                      %[[ARG:.*]]: !torch.optional<int>) -> !torch.none {
@mb.import_function
@torch.jit.script
def optional_arg(i: typing.Optional[int]) -> None:
    return


# CHECK-LABEL:   func.func @__torch__.calls_optional_arg(
# CHECK-SAME:                                       %[[ARG:.*]]: !torch.int) -> !torch.none {
# CHECK:           %[[CALLEE:.*]] = constant @__torch__.optional_arg : (!torch.optional<int>) -> !torch.none
# CHECK:           %[[DEREFINED:.*]] = torch.derefine %[[ARG]] : !torch.int to !torch.optional<int>
# CHECK:           %{{.*}} = call_indirect %[[CALLEE]](%[[DEREFINED]]) : (!torch.optional<int>) -> !torch.none
@mb.import_function
@torch.jit.script
def calls_optional_arg(i: int):
    optional_arg(i)


mb.module.operation.print()
print()
