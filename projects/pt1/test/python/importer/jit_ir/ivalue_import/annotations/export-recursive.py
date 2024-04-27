# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import typing

import torch
from torch_mlir.jit_ir_importer import ClassAnnotator, ModuleBuilder

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()


class Submodule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.exported = 1
        self.not_exported = 2

    def forward(self):
        return self.not_exported_method()

    def not_exported_method(self):
        return


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s = Submodule()

    def forward(self):
        return self.s.forward()


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)

annotator = ClassAnnotator()
class_type = recursivescriptmodule._c._type()
# CHECK-LABEL:   torch.class_type @__torch__.Submodule {
# CHECK:           torch.attr "exported" : !torch.int
# CHECK:           torch.attr private "not_exported" : !torch.int
# CHECK:           torch.method "forward", @{{.*}}
# CHECK:           torch.method private "not_exported_method", @{{.*}}
# CHECK:         }
annotator.exportNone(class_type)
annotator.exportPath(class_type, ["s", "exported"])
annotator.exportPath(class_type, ["s", "forward"])

# # TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c, annotator)
mb.module.operation.print()
