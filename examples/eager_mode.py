# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir.eager_mode.torch_mlir_tensor import TorchMLIRTensor

torch_a = torch.randn(5, requires_grad=True)
torch_b = torch.randn(5, requires_grad=True)

torch_c = torch_a + torch_b
torch_d = torch_a * torch_b
torch_e = torch_c / torch_d
torch_loss = torch_e.sum()
print("PyTorch loss: ", torch_loss)

torch_loss.backward()
print("PyTorch grad a: ", torch_a.grad)
print("PyTorch grad b: ", torch_b.grad)

a = TorchMLIRTensor(torch_a)
b = TorchMLIRTensor(torch_b)

c = a + b
d = a * b
e = c / d
loss = e.sum()
print("Torch-MLIR loss: ", loss)

loss.backward()
print("Torch-MLIR grad a: ", a.grad)
print("Torch-MLIR grad b: ", b.grad)
