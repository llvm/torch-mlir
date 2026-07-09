# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
import torch_mlir
import torch_mlir.fx


class MLPModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


module_to_compile = MLPModule()
# Fix the weights
torch.manual_seed(0)

backends = ["linalg-on-tensors", "tosa", "stablehlo"]

for backend in backends:
    print(f"Compiling MLPModule via FX to {backend}...")
    try:
        module = torch_mlir.fx.export_and_import(
            module_to_compile, torch.ones(2, 4), output_type=backend
        )
        print(f"Compilation to {backend} successful!")
        asm = module.operation.get_asm(large_elements_limit=10)
        lines = asm.splitlines()
        print(f"Generated ASM (first 10 lines of {backend}):")
        for line in lines[:10]:
            print("  ", line)
        print("-" * 40)
    except Exception as e:
        print(f"Compilation to {backend} FAILED!")
        raise e
