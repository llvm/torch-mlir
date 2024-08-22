# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

from typing import Dict, Optional

import torch
from torch_mlir.jit_ir_importer import ModuleBuilder

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d = {"key1": torch.tensor(1)}


# CHECK: torch.class_type @[[CLASSTYPE:.*]] {
# CHECK:         torch.attr "training" : !torch.bool
# CHECK:         torch.attr "_is_full_backward_hook" : !torch.optional<bool>
# CHECK:         torch.attr "d" : !torch.dict<str, tensor>
# CHECK:       }
# CHECK:       %[[K:.*]] = torch.constant.str "key1"
# CHECK:       %[[TENSOR:.*]] = torch.tensor.literal(dense<1> : tensor<si64>) : !torch.tensor<[],si64>
# CHECK:       %[[DICT:.*]] = torch.prim.DictConstruct
# CHECK-SAME     keys(%[[K]] : !torch.str) values(%[[TENSOR]] : !torch.tensor<[],si64>)
# CHECK-SAME:    -> !torch.dict<str, tensor>
# CHECK: torch.nn_module  {
# CHECK:           torch.slot "d", %[[DICT]] : !torch.dict<str, tensor>
# CHECK: } : !torch.nn.Module<"[[CLASSTYPE]]">

test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
