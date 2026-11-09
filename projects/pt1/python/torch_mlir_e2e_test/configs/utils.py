# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import Any

import ml_dtypes
import numpy as np
import torch


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    # NumPy has no native bfloat16, so reinterpret the bits as ml_dtypes.bfloat16.
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return t.numpy()


def numpy_to_tensor(a: np.ndarray) -> torch.Tensor:
    # torch.from_numpy does not accept ml_dtypes.bfloat16 arrays.
    if a.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(a.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(a)


def recursively_convert_to_numpy(o: Any):
    if isinstance(o, torch.Tensor):
        return tensor_to_numpy(o)
    if isinstance(o, tuple):
        return tuple(recursively_convert_to_numpy(x) for x in o)
    if isinstance(o, list):
        return [recursively_convert_to_numpy(x) for x in o]
    if isinstance(o, dict):
        return {k: recursively_convert_to_numpy(v) for k, v in o.items()}
    # No-op cases. Explicitly enumerated to avoid things sneaking through.
    if isinstance(o, str):
        return o
    if isinstance(o, float):
        return o
    if isinstance(o, int):
        return o
    raise Exception(f"Unexpected Python function input: {o}")


def recursively_convert_from_numpy(o: Any):
    if isinstance(o, np.ndarray):
        return numpy_to_tensor(o)
    if isinstance(o, tuple):
        return tuple(recursively_convert_from_numpy(x) for x in o)
    if isinstance(o, list):
        return [recursively_convert_from_numpy(x) for x in o]
    if isinstance(o, dict):
        return {k: recursively_convert_from_numpy(v) for k, v in o.items()}
    # No-op cases. Explicitly enumerated to avoid things sneaking through.
    if isinstance(o, str):
        return o
    if isinstance(o, float):
        return o
    if isinstance(o, int):
        return o
    raise Exception(f"Unexpected Python function output: {o}")
