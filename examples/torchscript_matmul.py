# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from PIL import Image
import requests
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend


mb = ModuleBuilder()

def predictions(torch_func, jit_func, data, output):
    golden_prediction = torch_func(data)
    print("PyTorch prediction")
    print(golden_prediction)
    prediction = torch.from_numpy(jit_func(data.numpy()))
    print("torch-mlir prediction")
    print(prediction)


class ModelModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 10)
        self.train(False)

    def forward(self, data):
        return self.fc(data)


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s = ModelModule()

    def forward(self, x):
        return self.s.forward(x)


data = torch.rand(4, 64)
output = ModelModule().forward(data)

test_module = TestModule()
class_annotator = ClassAnnotator()
recursivescriptmodule = torch.jit.script(test_module)
torch.jit.save(recursivescriptmodule, "/tmp/foo.pt")

class_annotator.exportNone(recursivescriptmodule._c._type())
class_annotator.exportPath(recursivescriptmodule._c._type(), ["forward"])
class_annotator.annotateArgs(
    recursivescriptmodule._c._type(),
    ["forward"],
    [
        None,
        ([4, 64], torch.float32, True),
    ],
)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c, class_annotator)

backend = refbackend.RefBackendLinalgOnTensorsBackend()
with mb.module.context:
    pm = PassManager.parse('torchscript-module-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline')
    pm.run(mb.module)

compiled = backend.compile(mb.module)
jit_module = backend.load(compiled)

predictions(test_module.forward, jit_module.forward, data, output)
