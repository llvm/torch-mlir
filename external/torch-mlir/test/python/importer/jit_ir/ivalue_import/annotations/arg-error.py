# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import typing

import torch
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
# RUN: %PYTHON %s | FileCheck %s

mb = ModuleBuilder()


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        return


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)

annotator = ClassAnnotator()
class_type = recursivescriptmodule._c._type()
try:
    annotator.annotateArgs(class_type, [], [])
except Exception as e:
    # CHECK: Empty annotated path. Can only annotate shapes/dtypes of a method of a class.
    print(e)

try:
    annotator.annotateArgs(class_type, ['forward'], [None])
except Exception as e:
    # CHECK: Arg annotations should have one entry per function parameter (including self).
    print(e)

try:
    annotator.annotateArgs(class_type, ['forward'], [None, ([3, 4], 42, False)])
except Exception as e:
    # This is just the raw repr of the object in quotes.
    # CHECK: unsupported scalar type '42'
    print(e)

# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c, annotator)
