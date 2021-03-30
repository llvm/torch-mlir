# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import typing

import torch
import torch_mlir

# RUN: %PYTHON %s | FileCheck %s

mb = torch_mlir.ModuleBuilder()


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        return


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)

annotator = torch_mlir.ClassAnnotator()
class_type = recursivescriptmodule._c._type()
try:
    annotator.annotateShapesAndDtypes(class_type, [], [])
except Exception as e:
    # CHECK: Empty annotated path. Can only annotate shapes/dtypes of a method of a class.
    print(e)

try:
    annotator.annotateShapesAndDtypes(class_type, ['forward'], [None])
except Exception as e:
    # CHECK: Arg annotations should have one entry per function parameter (including self).
    print(e)

try:
    annotator.annotateShapesAndDtypes(class_type, ['forward'], [None, ([3, 4], 42)])
except Exception as e:
    # This is just the raw repr of the object in quotes.
    # CHECK: unsupported scalar type '42'
    print(e)

# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c, annotator)
