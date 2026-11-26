# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch
import torch.nn as nn

from torch_mlir import fx
from torch_mlir.extras.fx_importer import TORCH_DTYPE_TO_INT


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# CHECK-LABEL: test_complex_dtype_int_codes
# CHECK: complex32=8 complex64=9 complex128=10
def test_complex_dtype_int_codes():
    assert TORCH_DTYPE_TO_INT[torch.complex32] == 8
    assert TORCH_DTYPE_TO_INT[torch.complex64] == 9
    assert TORCH_DTYPE_TO_INT[torch.complex128] == 10
    print(
        f"complex32={TORCH_DTYPE_TO_INT[torch.complex32]} "
        f"complex64={TORCH_DTYPE_TO_INT[torch.complex64]} "
        f"complex128={TORCH_DTYPE_TO_INT[torch.complex128]}"
    )


@run
# CHECK-LABEL: test_import_complex64
# CHECK: %[[C9:.+]] = torch.constant.int 9
# CHECK: torch.prims.convert_element_type %{{.+}}, %[[C9]] {{.*}}-> !torch.vtensor<[4],complex<f32>>
def test_import_complex64():
    class Model(nn.Module):
        def forward(self, x):
            return x.to(torch.complex64)

    print(fx.export_and_import(Model(), torch.randn(4)))


@run
# CHECK-LABEL: test_import_complex128
# CHECK: %[[C10:.+]] = torch.constant.int 10
# CHECK: torch.prims.convert_element_type %{{.+}}, %[[C10]] {{.*}}-> !torch.vtensor<[4],complex<f64>>
def test_import_complex128():
    class Model(nn.Module):
        def forward(self, x):
            return x.to(torch.complex128)

    print(fx.export_and_import(Model(), torch.randn(4)))
