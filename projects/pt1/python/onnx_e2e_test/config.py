# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""
Golden-vs-backend evaluation for ONNX-native end-to-end tests.

Provides two functions:
  - run_golden: evaluate an onnx.ModelProto (ONNX Runtime for Microsoft contrib ops)
  - run_backend: compile the same ModelProto through a torch-mlir backend

Both return a list of torch.Tensor so the existing torch_mlir_e2e_test
reporting/compare layer (torch.allclose) can be reused.
"""

from typing import List

import numpy as np
import onnx
import torch

from torch_mlir.compiler_utils import (
    OutputType,
    TensorPlaceholder,
)
from torch_mlir.extras import onnx_importer
from torch_mlir.dialects import torch as torch_d
from torch_mlir.ir import Context
from torch_mlir_e2e_test.configs.onnx_backend import module_lowering
from torch_mlir_e2e_test.configs.utils import recursively_convert_from_numpy


from .framework import _INPUT_RANGE_ATTR
from .reference import create_reference_evaluator

# Torch-MLIR execution backends
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import (
    RefBackendLinalgOnTensorsBackend,
)
from torch_mlir_e2e_test.tosa_backends.linalg_on_tensors import (
    LinalgOnTensorsTosaBackend,
)

# ---------------------------------------------------------------------------
# Input materialisation
# ---------------------------------------------------------------------------

_DYNAMIC_DIM_PLACEHOLDER = 2


def _numpy_dtype_for_torch(dtype: torch.dtype) -> np.dtype:
    """Map a torch.dtype to a numpy dtype for seeding input arrays."""
    try:
        return torch.empty((), dtype=dtype).numpy().dtype
    except TypeError as e:
        raise ValueError(
            f"Unsupported torch dtype for input generation: {dtype}"
        ) from e


def _is_integer_dtype(np_dtype: np.dtype) -> bool:
    return np.issubdtype(np_dtype, np.integer) or np_dtype == np.bool_


def materialize_inputs(
    inputs: List[TensorPlaceholder],
    seed: int = 0,
) -> List[np.ndarray]:
    """Materialise seeded numpy arrays from a list of TensorPlaceholder specs.

    Dynamic dimensions (shape element == -1) are replaced with
    ``_DYNAMIC_DIM_PLACEHOLDER`` (2) so BOTH golden and SUT receive the same
    concrete tensors.

    A placeholder may carry an optional ``(low, high)`` range (set by
    ``@annotate_inputs``) overriding the default sampling range for float and
    integer inputs; bool inputs ignore it.

    Args:
        inputs: Ordered list of TensorPlaceholder input specs.
        seed: RNG seed (default 0) — MUST be the same for golden and SUT.

    Returns:
        A list of numpy arrays, one per placeholder, in the same order.
    """
    rng = np.random.default_rng(seed)
    arrays: List[np.ndarray] = []
    for placeholder in inputs:
        shape = tuple(
            d if d >= 0 else _DYNAMIC_DIM_PLACEHOLDER for d in placeholder.shape
        )
        np_dtype = _numpy_dtype_for_torch(placeholder.dtype)
        value_range = getattr(placeholder, _INPUT_RANGE_ATTR, None)
        # Default to torch_mlir_e2e_test.TestUtils defaults: floats uniform
        # [0, 1), ints uniform [0, 10), bools [0, 2).
        if np_dtype == np.bool_:
            arr = rng.integers(0, 2, size=shape, dtype=np.bool_)
        elif _is_integer_dtype(np_dtype):
            low, high = value_range if value_range is not None else (0, 10)
            arr = rng.integers(low, high, size=shape, dtype=np_dtype)
        else:
            low, high = value_range if value_range is not None else (0.0, 1.0)
            arr = rng.uniform(low, high, size=shape).astype(np_dtype)
        arrays.append(arr)
    return arrays


# ---------------------------------------------------------------------------
# Golden evaluation via the shared reference evaluator
# ---------------------------------------------------------------------------


def run_golden(
    model_proto: onnx.ModelProto,
    numpy_inputs: List[np.ndarray],
) -> List[torch.Tensor]:
    """Evaluate *model_proto*, using ONNX Runtime for Microsoft contrib ops.

    Args:
        model_proto: The ONNX model.
        numpy_inputs: Concrete input arrays, one per graph input, in order.

    Returns:
        A list of torch.Tensor outputs.
    """
    input_names = [inp.name for inp in model_proto.graph.input]
    feed = {name: arr for name, arr in zip(input_names, numpy_inputs)}
    sess = create_reference_evaluator(model_proto)
    raw_outputs = sess.run(None, feed)
    return [recursively_convert_from_numpy(out) for out in raw_outputs]


# ---------------------------------------------------------------------------
# SUT evaluation via a torch-mlir Linalg backend
# ---------------------------------------------------------------------------


def _uses_com_microsoft_domain(raw_model: onnx.ModelProto) -> bool:
    return any(opset.domain == "com.microsoft" for opset in raw_model.opset_import)


def _import_onnx_for_e2e(serialized: bytes):
    """Imports a serialized ONNX model into MLIR.

    For models that use the com.microsoft domain, this skips
    onnx.shape_inference.infer_shapes: ONNX's shape inference does not
    understand com.microsoft contrib ops and clears the type information
    on their outputs. Such models (built by `OnnxTestCase`) already carry
    complete value_info for every tensor (inferred in OnnxTestCase.build),
    so shape inference not needed for them. This is fine for single-op
    test graphs but may need refinement for larger mixed models.
    """
    raw_model = onnx.load_from_string(serialized)
    if not _uses_com_microsoft_domain(raw_model):
        raw_model = onnx.shape_inference.infer_shapes(raw_model, data_prop=True)
    context = Context()
    torch_d.register_dialect(context)
    model_info = onnx_importer.ModelInfo(raw_model)
    m = model_info.create_module(context=context)
    imp = onnx_importer.NodeImporter.define_function(model_info.main_graph, m.operation)
    imp.import_all()
    return m


def run_backend(
    model_proto: onnx.ModelProto,
    numpy_inputs: List[np.ndarray],
    output_type: str = "linalg-on-tensors",
    backend=None,
) -> List[torch.Tensor]:
    """Compile *model_proto* through a torch-mlir backend and run it.

    The ModelProto is serialised, imported via ``import_onnx``, lowered from
    the ONNX dialect to the Torch backend contract, then to *output_type*
    (``"linalg-on-tensors"`` or ``"tosa"``).  The resulting module is compiled
    and executed through *backend* (a ``compile``/``load`` backend whose
    ``load`` returns an object exposing a ``main_graph`` invoker).  Both the
    Linalg and the TOSA backend ultimately execute on linalg via RefBackend,
    so numeric results are directly comparable to the golden.

    Args:
        model_proto: The ONNX model.
        numpy_inputs: Concrete input arrays matching the model inputs.
        output_type: torch-mlir OutputType to lower to.
        backend: Backend instance; defaults to the Linalg backend.

    Returns:
        A list of torch.Tensor outputs.
    """
    if backend is None:
        backend = RefBackendLinalgOnTensorsBackend()

    # Serialise once.
    serialized = model_proto.SerializeToString()

    # Import to MLIR.
    mlir_module = _import_onnx_for_e2e(serialized)

    # Lower ONNX dialect -> Torch backend contract -> requested backend dialect,
    # reusing the shared lowering sequence from the torch-level ONNX e2e config.
    backend_module = module_lowering(
        verbose=False,
        output_type=OutputType.get(output_type),
        torch_mod=mlir_module,
    )

    # Compile and run.
    compiled = backend.compile(backend_module)
    invoker = backend.load(compiled)

    raw_outputs = invoker.main_graph(*numpy_inputs)

    # Normalise: a backend may return a single ndarray or a tuple.
    if isinstance(raw_outputs, np.ndarray):
        raw_outputs = (raw_outputs,)

    outputs = [recursively_convert_from_numpy(out) for out in raw_outputs]

    # RefBackend's raw memref buffers don't carry 8-bit signedness, so an
    # op with result type of ui8 comes back from the compiled
    # artifact as a torch.int8. Reinterpret based on the ONNX graph's
    # declared output dtype.
    output_dtypes = [
        onnx.helper.tensor_dtype_to_np_dtype(out.type.tensor_type.elem_type)
        for out in model_proto.graph.output
    ]
    outputs = [
        _reinterpret_to_dtype(out, np_dtype)
        for out, np_dtype in zip(outputs, output_dtypes)
    ]

    return outputs


def _reinterpret_to_dtype(output: torch.Tensor, np_dtype: np.dtype) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        return output
    target_dtype = {np.dtype(np.uint8): torch.uint8, np.dtype(np.int8): torch.int8}.get(
        np.dtype(np_dtype)
    )
    if target_dtype is None or output.dtype == target_dtype:
        return output
    if output.dtype in (torch.uint8, torch.int8) and target_dtype in (
        torch.uint8,
        torch.int8,
    ):
        return output.view(target_dtype)
    return output


# ---------------------------------------------------------------------------
# Test configs — pair the reference evaluator with a SUT backend
# ---------------------------------------------------------------------------


class OnnxLinalgTestConfig:
    """Golden reference + SUT lowered to linalg-on-tensors on the Linalg backend.

    Usage::

        cfg = OnnxLinalgTestConfig()
        golden_outputs = cfg.run_golden(model_proto, numpy_inputs)
        sut_outputs = cfg.run_backend(model_proto, numpy_inputs)
        # compare with torch.allclose …
    """

    def run_golden(
        self,
        model_proto: onnx.ModelProto,
        numpy_inputs: List[np.ndarray],
    ) -> List[torch.Tensor]:
        return run_golden(model_proto, numpy_inputs)

    def run_backend(
        self,
        model_proto: onnx.ModelProto,
        numpy_inputs: List[np.ndarray],
    ) -> List[torch.Tensor]:
        return run_backend(model_proto, numpy_inputs)


class OnnxTosaBackendTestConfig:
    """Golden reference + SUT lowered to TOSA, executed via linalg.

    Lowers the imported model to the ``tosa`` dialect, then compiles it with
    ``LinalgOnTensorsTosaBackend`` (TOSA -> linalg -> RefBackend) so numeric
    results are comparable to the golden.
    """

    def run_golden(
        self,
        model_proto: onnx.ModelProto,
        numpy_inputs: List[np.ndarray],
    ) -> List[torch.Tensor]:
        return run_golden(model_proto, numpy_inputs)

    def run_backend(
        self,
        model_proto: onnx.ModelProto,
        numpy_inputs: List[np.ndarray],
    ) -> List[torch.Tensor]:
        return run_backend(
            model_proto,
            numpy_inputs,
            output_type="tosa",
            backend=LinalgOnTensorsTosaBackend(),
        )
