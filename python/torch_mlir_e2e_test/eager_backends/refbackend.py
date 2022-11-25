# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch

from torch_mlir.compiler_utils import (
    get_module_name_for_debug_dump,
    run_pipeline_with_repro_report,
)
from torch_mlir.eager_mode.torch_mlir_eager_backend import (
    TorchMLIREagerBackend,
    TensorMetaData,
)
from torch_mlir.ir import Module
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import (
    RefBackendLinalgOnTensorsBackend,
)

NUMPY_TO_TORCH_DTYPE_DICT = {
    np.bool: torch.bool,
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}

_ref_backend = RefBackendLinalgOnTensorsBackend()


class EagerModeRefBackend(TorchMLIREagerBackend):
    """Main entry-point for the reference backend for eager mode.

    RefBackend uses numpy.ndarray representations of tensors and thus all of the wrapping and unwrapping
    and munging here is done to between torch.Tensor and numpy.ndarray.
    """

    module_to_refbackend_invoker = {}

    def get_torch_metadata(
        self, tensor: np.ndarray, kwargs: Dict[str, Any]
    ) -> TensorMetaData:
        return TensorMetaData(
            size=tensor.shape,
            dtype=NUMPY_TO_TORCH_DTYPE_DICT[tensor.dtype.type],
            requires_grad=tensor.dtype in {np.float, np.float32, np.float64}
            and kwargs.get("requires_grad", False),
        )

    def compile(self, imported_module: Module):
        """Lower the imported TS module to linalg and then further compile for the reference backend and then call."""
        fn_name = get_module_name_for_debug_dump(imported_module)
        module_hash = str(imported_module)
        if module_hash not in self.module_to_refbackend_invoker:
            run_pipeline_with_repro_report(
                imported_module,
                "builtin.module(torch-function-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline)",
                "EagerMode",
            )
            self.module_to_refbackend_invoker[module_hash] = _ref_backend.load(
                _ref_backend.compile(imported_module)
            )

        ref_backend_invoker = self.module_to_refbackend_invoker[module_hash]
        op_mlir_backend_callable = getattr(ref_backend_invoker, fn_name)
        assert (
            op_mlir_backend_callable is not None
        ), f"Couldn't find function in module."
        return op_mlir_backend_callable

    def copy_into(self, dst: np.ndarray, src: np.ndarray):
        np.copyto(dst, src)

    def transfer_from_device_to_torch(self, e: np.ndarray):
        return torch.from_numpy(e).clone()

    def transfer_from_torch_to_device(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().numpy()
