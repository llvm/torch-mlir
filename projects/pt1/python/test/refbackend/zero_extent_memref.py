# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import numpy as np

from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import (
    _canonicalize_zero_element_strides,
)


def print_element_strides(shape):
    array = np.empty(shape, dtype=np.float32)
    array = np.lib.stride_tricks.as_strided(
        array, shape=shape, strides=(0,) * len(shape)
    )
    canonical = _canonicalize_zero_element_strides(array)
    strides = tuple(stride // canonical.itemsize for stride in canonical.strides)
    print(f"{shape}: {strides}")


print_element_strides((0,))
# CHECK: (0,): (1,)

print_element_strides((0, 10, 3))
# CHECK: (0, 10, 3): (30, 3, 1)

print_element_strides((5, 0, 0))
# CHECK: (5, 0, 0): (0, 0, 1)

print_element_strides((1, 0, 3))
# CHECK: (1, 0, 3): (0, 3, 1)

nonempty = np.empty((2, 3), dtype=np.float32)
print(_canonicalize_zero_element_strides(nonempty) is nonempty)
# CHECK: True
