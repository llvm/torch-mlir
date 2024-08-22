# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s

"""This file exists so that the tests can find/configure torch_mlir.

It allows the test file to be standalone and used verbatim in other
projects (i.e. by just providing this file on the side).
"""

from torch_mlir import ir
from torch_mlir.extras import onnx_importer


def configure_context(context):
    from torch_mlir.dialects import torch as torch_d

    torch_d.register_dialect(context)
