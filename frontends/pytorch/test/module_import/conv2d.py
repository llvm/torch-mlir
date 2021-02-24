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

        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3))

    def forward(self, x):
        return self.conv1(x)

# CHECK-LABEL:   torch.class_type
# CHECK-SAME:        @[[CLASSTYPE:.*]] {
# CHECK:           torch.method "forward", @[[SYMNAME:.*]]
# CHECK:         }

# CHECK-LABEL:   func private
# CHECK-SAME:                 @[[SYMNAME]](
# CHECK-SAME:                                 %[[SELF:.*]]: !torch.nn.Module<"[[CLASSTYPE]]">,
# CHECK-SAME:                                 %[[INPUT:.*]]: !numpy.ndarray<*:!numpy.any_dtype>)

# TODO(brycearden): Check's could be improved here, but this will do for now
# CHECK: torch.method "_conv_forward"
# CHECK: torch.kernel_call "aten::conv2d"

test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()