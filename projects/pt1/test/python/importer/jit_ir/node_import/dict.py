# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import torch
from torch_mlir.jit_ir_importer import ModuleBuilder
import collections
from typing import Tuple, Optional, List, NamedTuple, Dict

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()


# CHECK-LABEL:   func.func @__torch__.dict_literal_empty() -> !torch.dict<str, tensor> {
# CHECK:           %[[DICT:.*]] = torch.prim.DictConstruct keys() values() -> !torch.dict<str, tensor>
# CHECK:           return %[[DICT]] : !torch.dict<str, tensor>
@mb.import_function
@torch.jit.script
def dict_literal_empty() -> Dict[str, torch.Tensor]:
    return {}


# CHECK-LABEL:   func.func @__torch__.dict_literal(
# CHECK-SAME:        %[[K0:.*]]: !torch.str, %[[V0:.*]]: !torch.tensor,
# CHECK-SAME:        %[[K1:.*]]: !torch.str, %[[V1:.*]]: !torch.tensor)
# CHECK-SAME:        -> !torch.dict<str, optional<tensor>> {
# CHECK:           %[[DICT:.*]] = torch.prim.DictConstruct
# CHECK-SAME:        keys(%[[K0]], %[[K1]] : !torch.str, !torch.str)
# CHECK-SAME:        values(%[[V0]], %[[V1]] : !torch.tensor, !torch.tensor) ->
# CHECK-SAME:        !torch.dict<str, optional<tensor>>
# CHECK:           return %[[DICT]] : !torch.dict<str, optional<tensor>>
# CHECK:         }
@mb.import_function
@torch.jit.script
def dict_literal(k0: str, v0, k1: str, v1) -> Dict[str, Optional[torch.Tensor]]:
    my_dict: Dict[str, Optional[torch.Tensor]] = {k0: v0, k1: v1}
    return my_dict


mb.module.operation.print()
print()
