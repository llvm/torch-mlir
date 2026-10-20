# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""Shared helpers for ONNX-native end-to-end tests."""

import numpy as np
import torch


def numpy_dtype_for_torch(dtype: torch.dtype) -> np.dtype:
    """Map a torch.dtype to the equivalent numpy dtype."""
    try:
        return torch.empty((), dtype=dtype).numpy().dtype
    except TypeError as e:
        raise ValueError(
            f"Unsupported torch dtype for numpy conversion: {dtype}"
        ) from e
