# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import typing

import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return

test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)

annotator = torch_mlir.ClassAnnotator()
class_type = recursivescriptmodule._c._type()
# CHECK: func private @__torch__.TestModule.forward(
# CHECK-SAME: %arg0: !torch.nn.Module<"__torch__.TestModule">,
# CHECK-SAME: %arg1: !torch.tensor {torch.type_bound = !torch.vtensor<[?,1024],si8>},
# CHECK-SAME: %arg2: !torch.tensor {torch.type_bound = !torch.vtensor<[],f32>}
# CHECK-SAME: ) -> !torch.none
annotator.annotateArgs(class_type, ['forward'], [
    None,
    ((-1, 1024), torch.int8, True),
    ((), torch.float, True),
])

# # TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c, annotator)
mb.module.operation.print()
