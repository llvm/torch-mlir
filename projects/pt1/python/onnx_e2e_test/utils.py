# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""Shared helpers for ONNX-native end-to-end tests."""

from typing import List

import numpy as np
import onnx
import onnx.helper
import torch


def numpy_dtype_for_torch(dtype: torch.dtype) -> np.dtype:
    """Map a torch.dtype to the equivalent numpy dtype."""
    try:
        return torch.empty((), dtype=dtype).numpy().dtype
    except TypeError as e:
        raise ValueError(
            f"Unsupported torch dtype for numpy conversion: {dtype}"
        ) from e


def cast_outputs_to_declared_dtypes(
    model_proto: onnx.ModelProto, arrays: List[np.ndarray]
) -> List[np.ndarray]:
    """Reinterpret backend outputs as the dtypes declared by the model given
    that backends lower to signless integers (e.g., a ui8 result is
    returned as int8).
    """
    result = []
    for info, arr in zip(model_proto.graph.output, arrays):
        if isinstance(arr, np.ndarray) and info.type.HasField("tensor_type"):
            declared = np.dtype(
                onnx.helper.tensor_dtype_to_np_dtype(info.type.tensor_type.elem_type)
            )
            if (
                arr.dtype != declared
                and np.issubdtype(arr.dtype, np.integer)
                and np.issubdtype(declared, np.integer)
                and arr.dtype.itemsize == declared.itemsize
            ):
                arr = arr.view(declared)
        result.append(arr)
    # Keep any extra outputs that have no declared counterpart.
    result.extend(arrays[len(result) :])
    return result
