# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import Any

import numpy as np
import torch


def recursively_convert_to_numpy(o: Any):
    if isinstance(o, torch.Tensor):
        return o.numpy()
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
        return torch.from_numpy(o)
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
