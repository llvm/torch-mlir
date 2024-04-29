# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import typing

import torch
from torch_mlir.jit_ir_importer import ClassAnnotator, ModuleBuilder

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)

annotator = ClassAnnotator()
class_type = recursivescriptmodule._c._type()
# CHECK: func.func private @__torch__.TestModule.forward(
# CHECK-SAME: %arg0: !torch.nn.Module<"__torch__.TestModule">,
# CHECK-SAME: %arg1: !torch.tensor {torch.type_bound = !torch.vtensor<[?,1024],si8>},
# CHECK-SAME: %arg2: !torch.tensor {torch.type_bound = !torch.vtensor<[],f32>}
# CHECK-SAME: ) -> !torch.none
annotator.annotateArgs(
    class_type,
    ["forward"],
    [
        None,
        ((-1, 1024), torch.int8, True),
        ((), torch.float, True),
    ],
)

# # TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c, annotator)
mb.module.operation.print()
