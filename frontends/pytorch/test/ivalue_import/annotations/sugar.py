# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

# RUN: %PYTHON %s | FileCheck %s

import torch

import torch_mlir
from torch_mlir.torchscript.annotations import (
    annotate_args, export, extract_annotations
)

class MmModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    @export
    @annotate_args([
        None,
        ([3, 4], torch.float32),
        ([4, 5], torch.float32),
    ])
    def forward(self, lhs, rhs):
        return torch.mm(lhs, rhs)

module = MmModule()
annotator = torch_mlir.ClassAnnotator()
extract_annotations(module, torch.jit.script(module), annotator)
print(annotator)

# CHECK: ClassAnnotator {
# CHECK:   ClassAnnotation('__torch__.MmModule') {
# CHECK:     MethodAnnotation('forward') {
# CHECK:       isExported = true
# CHECK:       argAnnotations =
# CHECK:         ArgAnnotation(0) {
# CHECK:           dtype = <none>
# CHECK:           shape = <none>
# CHECK:         }
# CHECK:         ArgAnnotation(1) {
# CHECK:           dtype = Float
# CHECK:           shape = [3, 4]
# CHECK:         }
# CHECK:         ArgAnnotation(2) {
# CHECK:           dtype = Float
# CHECK:           shape = [4, 5]
# CHECK:         }
# CHECK:     }
# CHECK:   }
# CHECK: }
