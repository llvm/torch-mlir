# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import typing

import torch
import torch_mlir

import npcomp
from npcomp.compiler.pytorch.backend import iree, frontend_lowering
from npcomp.compiler.utils import logging

import test_utils

logging.enable()

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.tanh(x)

test_module = TestModule()
class_annotator = torch_mlir.ClassAnnotator()
recursivescriptmodule = torch.jit.script(test_module)
torch.jit.save(recursivescriptmodule, '/tmp/foo.pt')

class_annotator.exportNone(recursivescriptmodule._c._type())
class_annotator.exportPath(recursivescriptmodule._c._type(), ['forward'])
class_annotator.annotateShapesAndDtypes(recursivescriptmodule._c._type(), ['forward'], [
    None,
    ([2, 3, -1], torch.float32)
])
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c, class_annotator)
#mb.module.operation.print()

backend = iree.CompilerBackend()
compiled = backend.compile(frontend_lowering.lower_object_graph(mb.module))
jit_module = backend.load(compiled)

torch.manual_seed(0)
input = torch.rand(2, 3, 1)
test_utils.compare_outputs(test_module.forward, jit_module.forward, input)
