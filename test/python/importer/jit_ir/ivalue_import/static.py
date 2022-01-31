# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

from typing import Dict, Optional

import torch
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir.dialects.torch.importer.jit_ir.torchscript_annotations import extract_annotations
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export
from torch_mlir.passmanager import PassManager

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mp2d = torch.nn.MaxPool2d(kernel_size=[3, 3],
                                       stride=[2, 2],
                                       padding=[1, 1],
                                       dilation=[1, 1])

    @export
    @annotate_args([
        None,
        ([1, 64, 112, 112], torch.float32, True),
    ])
    def forward(self, x):
        return self.mp2d(x)


# CHECK-LABEL: func @forward
# CHECK:  %[[RES:.*]] = torch.aten.max_pool2d {{.*}} -> !torch.vtensor<[1,64,56,56],f32>
# CHECK:  return %[[RES:.*]] : !torch.vtensor<[1,64,56,56],f32>

test_module = TestModule()
mb = ModuleBuilder()
recursivescriptmodule = torch.jit.script(test_module)
class_annotator = ClassAnnotator()
extract_annotations(test_module, recursivescriptmodule, class_annotator)

mb.import_module(recursivescriptmodule._c, class_annotator)
with mb.module.context:
    pm = PassManager.parse("torchscript-module-to-torch-backend-pipeline")
    pm.run(mb.module)

mb.module.operation.print()
