# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import typing

import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()


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

annotator = torch_mlir.ClassAnnotator()
class_type = recursivescriptmodule._c._type()
# CHECK-LABEL:   torch.class_type @__torch__.Submodule {
# CHECK:           torch.attr "exported" : i64                                                                                                                                                                   
# CHECK:           torch.attr private "not_exported" : i64
# CHECK:           torch.method "forward", @{{.*}}
# CHECK:           torch.method private "not_exported_method", @{{.*}}
# CHECK:         }              
annotator.exportNone(class_type)
annotator.exportPath(['s', 'exported'], class_type)
annotator.exportPath(['s', 'forward'], class_type)

# # TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c, annotator)
mb.module.operation.print()
