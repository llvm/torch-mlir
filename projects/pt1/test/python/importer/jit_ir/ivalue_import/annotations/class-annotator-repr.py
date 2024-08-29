# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import typing

import torch
from torch_mlir.jit_ir_importer import ClassAnnotator, ModuleBuilder

# RUN: %PYTHON %s | FileCheck %s

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

    def forward(self, tensor, value_tensor):
        return self.s.forward()


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)

annotator = ClassAnnotator()
class_type = recursivescriptmodule._c._type()

annotator.exportNone(class_type)
annotator.exportPath(class_type, ["s", "exported"])
annotator.exportPath(class_type, ["s", "forward"])
annotator.annotateArgs(
    class_type,
    ["forward"],
    [
        None,
        ((1024, 2), torch.float32, False),
        ((42, -1, 7), torch.int8, True),
    ],
)

# "Change detector" test + "documentation" for the repr of `ClassAnnotator`.
# This is semi-load-bearing because users interact with this class and repr
# will show up in error messages, so should be pretty readable.
#
# CHECK:      ClassAnnotator {
# CHECK-NEXT:   ClassAnnotation('__torch__.Submodule') {
# CHECK-NEXT:     AttributeAnnotation('training') {
# CHECK-NEXT:       isExported = false
# CHECK-NEXT:     }
# CHECK-NEXT:     AttributeAnnotation('_is_full_backward_hook') {
# CHECK-NEXT:       isExported = false
# CHECK-NEXT:     }
# CHECK-NEXT:     AttributeAnnotation('exported') {
# CHECK-NEXT:       isExported = true
# CHECK-NEXT:     }
# CHECK-NEXT:     AttributeAnnotation('not_exported') {
# CHECK-NEXT:       isExported = false
# CHECK-NEXT:     }
# CHECK-NEXT:     MethodAnnotation('forward') {
# CHECK-NEXT:       isExported = true
# CHECK-NEXT:       argAnnotations = <none>
# CHECK-NEXT:     }
# CHECK-NEXT:     MethodAnnotation('not_exported_method') {
# CHECK-NEXT:       isExported = false
# CHECK-NEXT:       argAnnotations = <none>
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT:   ClassAnnotation('__torch__.TestModule') {
# CHECK-NEXT:     AttributeAnnotation('training') {
# CHECK-NEXT:       isExported = false
# CHECK-NEXT:     }
# CHECK-NEXT:     AttributeAnnotation('_is_full_backward_hook') {
# CHECK-NEXT:       isExported = false
# CHECK-NEXT:     }
# CHECK-NEXT:     AttributeAnnotation('s') {
# CHECK-NEXT:       isExported = false
# CHECK-NEXT:     }
# CHECK-NEXT:     MethodAnnotation('forward') {
# CHECK-NEXT:       isExported = false
# CHECK-NEXT:       argAnnotations =
# CHECK-NEXT:         ArgAnnotation(0) {
# CHECK-NEXT:           dtype = <none>
# CHECK-NEXT:           shape = <none>
# CHECK-NEXT:           hasValueSemantics = false
# CHECK-NEXT:         }
# CHECK-NEXT:         ArgAnnotation(1) {
# CHECK-NEXT:           dtype = Float
# CHECK-NEXT:           shape = [1024, 2]
# CHECK-NEXT:           hasValueSemantics = false
# CHECK-NEXT:         }
# CHECK-NEXT:         ArgAnnotation(2) {
# CHECK-NEXT:           dtype = Char
# CHECK-NEXT:           shape = [42, -1, 7]
# CHECK-NEXT:           hasValueSemantics = true
# CHECK-NEXT:         }
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }

print(annotator)
