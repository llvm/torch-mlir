# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE for license information.

import torch
import torch_mlir
import collections
from typing import Tuple, Optional, NamedTuple

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()
NT = NamedTuple('NT', [('f1', Optional[torch.Tensor]),
                       ('f2', Optional[torch.Tensor])])

# CHECK-LABEL:   func @__torch__.tuple(
# CHECK-SAME:            %[[T0:.*]]: !torch.tensor,
# CHECK-SAME:            %[[T1:.*]]: !torch.tensor) ->
# CHECK-SAME:            !torch.tuple<!torch.tensor, !torch.tensor> {
# CHECK:           %[[RET:.*]] = torch.prim.TupleConstruct %[[T0]], %[[T1]] :
# CHECK-SAME:            !torch.tensor, !torch.tensor -> !torch.tuple<!torch.tensor, !torch.tensor>
# CHECK:           return %[[RET]] : !torch.tuple<!torch.tensor, !torch.tensor>


@mb.import_function
@torch.jit.script
def tuple(t0, t1):
  return t0, t1


# CHECK-LABEL:   func @__torch__.tuple_optional(
# CHECK-SAME:            %[[T0:.*]]: !torch.tensor,
# CHECK-SAME:            %[[T1:.*]]: !torch.tensor) ->
# CHECK-SAME:            !torch.tuple<!torch.optional<!torch.tensor>, !torch.optional<!torch.tensor>> {
# CHECK:           %[[TNEW:.*]] = torch.prim.TupleConstruct %[[T0]], %[[T1]] :
# CHECK-SAME:           !torch.tensor, !torch.tensor -> !torch.tuple<!torch.tensor, !torch.tensor>
# CHECK:           %[[RET:.*]] = torch.derefine %[[TNEW]] :
# CHECK-SAME:           !torch.tuple<!torch.tensor, !torch.tensor> to
# CHECK-SAME:           !torch.tuple<!torch.optional<!torch.tensor>, !torch.optional<!torch.tensor>>
# CHECK:           return %[[RET]] : !torch.tuple<!torch.optional<!torch.tensor>, !torch.optional<!torch.tensor>>


@mb.import_function
@torch.jit.script
def tuple_optional(
    t0, t1) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
  return t0, t1


# CHECK-LABEL:   func @__torch__.namedtuple_optional(
# CHECK-SAME:            %[[T0:.*]]: !torch.tensor,
# CHECK-SAME:            %[[T1:.*]]: !torch.tensor) ->
# CHECK-SAME:            !torch.tuple<!torch.optional<!torch.tensor>, !torch.optional<!torch.tensor>> {
# CHECK:           %[[RET:.*]] = torch.prim.TupleConstruct %[[T0]], %[[T1]] :
# CHECK-SAME:            !torch.tensor, !torch.tensor ->
# CHECK-SAME:            !torch.tuple<!torch.optional<!torch.tensor>, !torch.optional<!torch.tensor>>
# CHECK:           return %[[RET]] : !torch.tuple<!torch.optional<!torch.tensor>, !torch.optional<!torch.tensor>>
# CHECK:         }
#
@mb.import_function
@torch.jit.script
def namedtuple_optional(
    t0, t1) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
  return NT(t0, t1)


mb.module.operation.print()
print()
