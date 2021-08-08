# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

from typing import Dict, Optional

import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()


class TestModule(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.d = {"key1": torch.tensor(1)}


# CHECK: torch.class_type @[[CLASSTYPE:.*]] {
# CHECK:         torch.attr "training" : !torch.bool
# CHECK:         torch.attr "_is_full_backward_hook" : !torch.optional<!torch.bool>
# CHECK:         torch.attr "d" : !torch.dict<!torch.str, !torch.tensor>
# CHECK:       }
# CHECK:       %[[K:.*]] = torch.constant.str "key1"
# CHECK:       %[[TENSOR:.*]] = torch.tensor.literal(dense<1> : tensor<si64>) : !torch.tensor<[],si64>
# CHECK:       %[[DICT:.*]] = torch.prim.DictConstruct(keys %[[K]] values %[[TENSOR]]) :
# CHECK-SAME:    (!torch.str, !torch.tensor<[],si64>) -> !torch.dict<!torch.str, !torch.tensor>
# CHECK: torch.nn_module  {
# CHECK:           torch.slot "d", %[[DICT]] : !torch.dict<!torch.str, !torch.tensor>
# CHECK: } : !torch.nn.Module<"[[CLASSTYPE]]">

test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
