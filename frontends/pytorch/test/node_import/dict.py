# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir
import collections
from typing import Tuple, Optional, List, NamedTuple, Dict

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()


# CHECK-LABEL:   builtin.func @__torch__.dict_empty() -> !torch.dict<!torch.str, !torch.tensor> {
# CHECK:           %[[DICT:.*]] = torch.prim.DictConstruct  : () -> !torch.dict<!torch.str, !torch.tensor>
# CHECK:           return %[[DICT]] : !torch.dict<!torch.str, !torch.tensor>
@mb.import_function
@torch.jit.script
def dict_empty() -> Dict[str, torch.Tensor]:
  return {}


# CHECK-LABEL:   builtin.func @__torch__.dict_no_empty(
# CHECK-SAME:        %[[K0:.*]]: !torch.str, %[[V0:.*]]: !torch.tensor,
# CHECK-SAME:        %[[K1:.*]]: !torch.str, %[[V1:.*]]: !torch.tensor)
# CHECK-SAME:        -> !torch.dict<!torch.str, !torch.optional<!torch.tensor>> {
# CHECK:           %[[DICT:.*]] = torch.prim.DictConstruct
# CHECK-SAME:        (keys %[[K0]], %[[K1]] values %[[V0]], %[[V1]]) :
# CHECK-SAME:        (!torch.str, !torch.str, !torch.tensor, !torch.tensor) ->
# CHECK-SAME:        !torch.dict<!torch.str, !torch.optional<!torch.tensor>>
# CHECK:           return %[[DICT]] : !torch.dict<!torch.str, !torch.optional<!torch.tensor>>
# CHECK:         }
@mb.import_function
@torch.jit.script
def dict_no_empty(k0: str, v0, k1: str,
                  v1) -> Dict[str, Optional[torch.Tensor]]:
  my_dict: Dict[str, Optional[torch.Tensor]] = {k0: v0, k1: v1}
  return my_dict


mb.module.operation.print()
print()
