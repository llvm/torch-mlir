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
        self.s = "foo"
# CHECK: torch.class_type @[[CLASSTYPE:.*]] {
# TODO: Don't lose element type.
# CHECK:   torch.attr "s" : !torch.str
# CHECK: }
# CHECK: %[[STR:.*]] = torch.constant.str "foo"
# CHECK: torch.nn_module  {
# CHECK:   torch.slot "s", %[[STR]] : !torch.str
# CHECK: } : !torch.nn.Module<"[[CLASSTYPE]]">


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
