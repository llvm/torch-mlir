# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch

import torch_mlir


class HasLargeGlobalTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.big = torch.ones(512, 1024)

    def forward(self):
        return self.big


# CHECK:       func.func @forward() -> !torch.vtensor<[512,1024],f32> {
# CHECK-NEXT:    %[[LITERAL:.*]] = torch.vtensor.external.literal(@big) : !torch.vtensor<[512,1024],f32>
# CHECK-NEXT:    return %[[LITERAL]] : !torch.vtensor<[512,1024],f32>
print(torch_mlir.compile(HasLargeGlobalTensorModule(), [],
                         use_external_references_if_numel_exceeds=16))

# CHECK:       func.func @forward() -> !torch.vtensor<[512,1024],f32> {
# CHECK-NEXT:    %[[LITERAL:.*]] = torch.vtensor.literal(dense<1.000000e+00> : tensor<512x1024xf32>) : !torch.vtensor<[512,1024],f32>
# CHECK-NEXT:    return %[[LITERAL]] : !torch.vtensor<[512,1024],f32>
print(torch_mlir.compile(HasLargeGlobalTensorModule(), []))
