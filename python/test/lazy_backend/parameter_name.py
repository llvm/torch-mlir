# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s


import torch
import torch._lazy

import torch_mlir.reference_lazy_backend._REFERENCE_LAZY_BACKEND as lazy_backend

from run_test import run_test

lazy_backend._initialize()

device = "lazy"


# CHECK: Parameter names:
# CHECK:     fc1.weight
# CHECK:     fc1.bias
# -----
# CHECK: PASS - test_parameter_name
@run_test
def test_parameter_name():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.fc1(x)

    model = Model().to(device)

    for name, tensor in model.state_dict().items():
        assert lazy_backend.set_parameter_name(tensor, name), f"Failed to set parameter name: {name}"

    inputs = torch.ones(2, 2, dtype=torch.float32).to(device)
    assert inputs.device.type == device

    outputs = model(inputs)

    torch._lazy.mark_step()

    print(lazy_backend.get_latest_computation().debug_string())
