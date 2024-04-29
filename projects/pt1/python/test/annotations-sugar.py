# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch

from torch_mlir_e2e_test.annotations import annotate_args, export
from torch_mlir.jit_ir_importer import ClassAnnotator
from torch_mlir.jit_ir_importer.torchscript_annotations import extract_annotations


class MmModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 4], torch.float32, False),
            ([4, 5], torch.float32, True),
        ]
    )
    def forward(self, lhs, rhs):
        return torch.mm(lhs, rhs)


module = MmModule()
annotator = ClassAnnotator()
extract_annotations(module, torch.jit.script(module), annotator)
print(annotator)

# CHECK: ClassAnnotator {
# CHECK:   ClassAnnotation('{{.*}}.MmModule') {
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
# CHECK:           hasValueSemantics = false
# CHECK:         }
# CHECK:         ArgAnnotation(2) {
# CHECK:           dtype = Float
# CHECK:           shape = [4, 5]
# CHECK:           hasValueSemantics = true
# CHECK:         }
# CHECK:     }
# CHECK:   }
# CHECK: }
