# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
import torchvision

import torch_mlir

resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.eval()

module = torch_mlir.compile(resnet18, torch.ones(1, 3, 224, 224), output_type=torch_mlir.OutputType.TORCH)
print("TORCH OutputType\n", module.operation.get_asm(large_elements_limit=10))
module = torch_mlir.compile(resnet18, torch.ones(1, 3, 224, 224), output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
print("LINALG_ON_TENSORS OutputType\n", module.operation.get_asm(large_elements_limit=10))
# TODO: Debug why this is so slow.
module = torch_mlir.compile(resnet18, torch.ones(1, 3, 224, 224), output_type=torch_mlir.OutputType.TOSA)
print("TOSA OutputType\n", module.operation.get_asm(large_elements_limit=10))
