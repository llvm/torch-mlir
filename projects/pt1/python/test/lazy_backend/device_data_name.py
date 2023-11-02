# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s


import torch
import torch._lazy

import torch_mlir._mlir_libs._REFERENCE_LAZY_BACKEND as lazy_backend

from run_test import run_test

lazy_backend._initialize()

device = "lazy"


# CHECK: 0 input tensors found
# -----
# CHECK: PASS - test_no_device_data_name
@run_test
def test_no_device_data_name():
    x = torch.tensor(1).to(device)
    y = torch.tensor(2).to(device)
    z = x + y
    torch._lazy.mark_step()


# CHECK: Input tensor: input_x
# CHECK: 1 input tensors found
# -----
# CHECK: PASS - test_device_data_name
@run_test
def test_device_data_name():
    x = torch.tensor(1).to(device)
    y = torch.tensor(2).to(device)

    lazy_backend.set_parameter_name(x, "input_x")

    z = x + y
    torch._lazy.mark_step()
