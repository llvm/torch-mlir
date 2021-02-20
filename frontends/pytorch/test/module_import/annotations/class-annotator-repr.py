# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import typing

import torch
import torch_mlir

# RUN: %PYTHON %s | FileCheck %s

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

annotator.exportNone(class_type)
annotator.exportPath(['s', 'exported'], class_type)
annotator.exportPath(['s', 'forward'], class_type)

# "Change detector" test + "documentation" for the repr of `ClassAnnotator`.
# This is semi-load-bearing because users interact with this class and repr
# will show up in error messages, so should be pretty readable.
# CHECK: ClassAnnotator {
# CHECK:   ClassAnnotation('__torch__.Submodule') {
# CHECK:     AttributeAnnotation('exported') {
# CHECK:       isExported = true
# CHECK:     }
# CHECK:     AttributeAnnotation('not_exported') {
# CHECK:       isExported = false
# CHECK:     }
# CHECK:     MethodAnnotation('forward') {
# CHECK:       isExported = true
# CHECK:     }
# CHECK:     MethodAnnotation('not_exported_method') {
# CHECK:       isExported = false
# CHECK:     }
# CHECK:   }
# CHECK:   ClassAnnotation('__torch__.TestModule') {
# CHECK:     AttributeAnnotation('s') {
# CHECK:       isExported = false
# CHECK:     }
# CHECK:     MethodAnnotation('forward') {
# CHECK:       isExported = false
# CHECK:     }
# CHECK:   }
# CHECK: }
print(annotator)
