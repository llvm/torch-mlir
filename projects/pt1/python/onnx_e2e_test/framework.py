# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""
End-to-end testing framework for ONNX-native tests.

A test in this framework is described by an ``OnnxTest`` — a stable name,
a zero-argument factory that builds an ``onnx.ModelProto``, and a list of
``TensorPlaceholder`` input specs.
"""

from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
import onnx
import onnx.helper

from torch_mlir.compiler_utils import TensorPlaceholder


# ---------------------------------------------------------------------------
# OnnxTest — the fundamental test descriptor
# ---------------------------------------------------------------------------


class OnnxTest(NamedTuple):
    """A description of an ONNX end-to-end test case.

    Attributes:
        unique_name: Stable identifier used for error reporting and registry
            deduplication.
        model_factory: Zero-argument callable that builds and returns an
            ``onnx.ModelProto``.  Only invoked when the runner executes the
            test, so imports stay cheap and ``--filter`` skips building
            unselected models (each build runs the reference evaluator).
        inputs: Ordered list of ``TensorPlaceholder`` instances describing
            the shape and dtype of each model input.  Step 3 (golden
            generation) materialises seeded numpy arrays from these specs.
    """

    unique_name: str
    model_factory: Callable[[], onnx.ModelProto]
    inputs: List[TensorPlaceholder]


# ---------------------------------------------------------------------------
# build_model — convenience helper
# ---------------------------------------------------------------------------

# Opset 17 / IR version 8: a broadly supported, stable baseline.  Single-sourced
# so the probe and final models can't declare mismatched versions.
_DEFAULT_OPSET = 17
_DEFAULT_IR_VERSION = 8


def build_model(
    nodes: list,
    inputs: list,
    outputs: list,
    initializers: Optional[list] = None,
    opset: int = _DEFAULT_OPSET,
    ir_version: int = _DEFAULT_IR_VERSION,
    validate: bool = True,
) -> onnx.ModelProto:
    """Build an ``onnx.ModelProto`` with the graph name hard-coded to
    ``"main_graph"``.

    The ``"main_graph"`` name is required because the RefBackend invocation
    calls the compiled symbol by that name.

    Args:
        nodes: List of ``onnx.NodeProto`` objects (from
            ``onnx.helper.make_node``).
        inputs: List of ``onnx.ValueInfoProto`` objects describing graph
            inputs (from ``onnx.helper.make_tensor_value_info``).
        outputs: List of ``onnx.ValueInfoProto`` objects describing graph
            outputs.
        initializers: Optional list of ``onnx.TensorProto`` weight tensors.
        opset: ONNX opset version (default ``_DEFAULT_OPSET``).
        ir_version: ONNX IR version (default ``_DEFAULT_IR_VERSION``).
        validate: Run ``onnx.checker.check_model`` (default True).  Set False
            for the pass-1 probe model, whose outputs are intentionally
            shapeless (the checker requires a shape).

    Returns:
        An ``onnx.ModelProto`` (validated unless ``validate`` is False).
    """
    graph = onnx.helper.make_graph(
        nodes=nodes,
        name="main_graph",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers or [],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", opset)],
        ir_version=ir_version,
    )
    if validate:
        onnx.checker.check_model(model)
    return model


# ---------------------------------------------------------------------------
# Class-based authoring API (mirrors the torch e2e suite's Module + decorators)
# ---------------------------------------------------------------------------

# Maps a torch.dtype to its ONNX TensorProto element type, so an input is
# described exactly once (as a torch.dtype) and reused for both the ONNX
# value_info and the runtime TensorPlaceholder.
_TORCH_TO_ONNX_DTYPE = {
    torch.float16: onnx.TensorProto.FLOAT16,
    # bfloat16 omitted: onnx.reference can't evaluate bf16 arithmetic and
    # np_dtype_to_tensor_dtype can't round-trip ml_dtypes.bfloat16.
    torch.float32: onnx.TensorProto.FLOAT,
    torch.float64: onnx.TensorProto.DOUBLE,
    torch.int8: onnx.TensorProto.INT8,
    torch.int16: onnx.TensorProto.INT16,
    torch.int32: onnx.TensorProto.INT32,
    torch.int64: onnx.TensorProto.INT64,
    torch.uint8: onnx.TensorProto.UINT8,
    torch.bool: onnx.TensorProto.BOOL,
}

# Attribute the @annotate_inputs decorator stashes on graph(); read by
# OnnxTestCase to build the input value_infos and TensorPlaceholders.
_INPUT_ANNOTATIONS_ATTR = "_onnx_input_annotations"

# Type of a single input annotation: (name, torch.dtype, shape) with an
# optional 4th element (low, high) giving the value range to sample from.
InputAnnotation = Tuple[str, torch.dtype, List[int]]

# Attribute name under which input_placeholders() stashes an input's optional
# (low, high) range on its TensorPlaceholder; read by materialize_inputs.
_INPUT_RANGE_ATTR = "_onnx_input_range"


def annotate_inputs(annotations: List[InputAnnotation]):
    """Declare the runtime inputs of a test's ``graph`` method.

    Each entry is ``(name, dtype, shape)`` or, to override the default sampling
    range, ``(name, dtype, shape, (low, high))``.  The name is the ONNX tensor
    name used inside the graph; dtype/shape are written **once** here and reused
    to build both the ONNX input ``value_info`` and the runtime
    ``TensorPlaceholder``.  ``(low, high)`` mirrors the ``low``/``high``
    arguments authors pass to ``TestUtils.rand``/``randint`` in the torch e2e
    suite; when omitted, the framework default range is used.
    """

    def decorator(fn):
        setattr(fn, _INPUT_ANNOTATIONS_ATTR, annotations)
        return fn

    return decorator


class OnnxTestCase:
    """Base class for class-based ONNX e2e tests.

    Subclass it, decorate ``graph`` with ``@annotate_inputs`` and register the
    class with ``@register_onnx_test``.  The subclass name is the test's
    ``unique_name``.  ``graph`` returns ``(nodes, output_names[, initializers])``
    where ``output_names`` is the list of graph output tensor names.  The base
    infers each output's shape and dtype by running the model once through
    ``onnx.reference`` — authors never declare output shapes.  Example::

        @register_onnx_test
        class OnnxAdd_f32_basic(OnnxTestCase):
            @annotate_inputs([
                ("x", torch.float32, [4, 4]),
                ("y", torch.float32, [4, 4]),
            ])
            def graph(self):
                return [onnx.helper.make_node("Add", ["x", "y"], ["z"])], ["z"]
    """

    def graph(self):
        raise NotImplementedError

    @classmethod
    def _input_annotations(cls) -> List[InputAnnotation]:
        annotations = getattr(cls.graph, _INPUT_ANNOTATIONS_ATTR, None)
        if annotations is None:
            raise ValueError(
                f"{cls.__name__}.graph must be decorated with @annotate_inputs"
            )
        return annotations

    @classmethod
    def input_placeholders(cls) -> List[TensorPlaceholder]:
        placeholders = []
        for annotation in cls._input_annotations():
            _name, dtype, shape = annotation[:3]
            placeholder = TensorPlaceholder(shape, dtype)
            if len(annotation) > 3:
                setattr(placeholder, _INPUT_RANGE_ATTR, annotation[3])
            placeholders.append(placeholder)
        return placeholders

    @classmethod
    def build(cls) -> onnx.ModelProto:
        # Local imports so that the pure descriptor path (registration) does not
        # pull in the reference evaluator / numpy input machinery.
        import onnx.reference

        from .config import materialize_inputs

        annotations = cls._input_annotations()
        # A -1 in an annotation shape marks a dynamic dim: TensorPlaceholder /
        # materialize_inputs read it as -1 (substituting a concrete size at
        # runtime), but ONNX expects None for a symbolic dim -- a literal -1
        # would import as a static dim of -1 instead of a `?`.
        input_infos = [
            onnx.helper.make_tensor_value_info(
                name,
                _TORCH_TO_ONNX_DTYPE[dtype],
                [d if d >= 0 else None for d in shape],
            )
            for (name, dtype, shape, *_range) in annotations
        ]

        instance = cls()
        result = instance.graph()
        if len(result) == 3:
            nodes, output_names, initializers = result
        else:
            nodes, output_names = result
            initializers = None

        # Pass 1: build a model with shapeless outputs and run onnx.reference to
        # learn each output's concrete shape and dtype.  This model is not
        # validated (check_model requires a shape); it exists only to be run.
        probe_outputs = [
            onnx.helper.make_empty_tensor_value_info(name) for name in output_names
        ]
        probe_model = build_model(
            nodes, input_infos, probe_outputs, initializers=initializers, validate=False
        )
        numpy_inputs = materialize_inputs(cls.input_placeholders(), seed=0)
        feed = {
            annotation[0]: arr for (annotation, arr) in zip(annotations, numpy_inputs)
        }
        golden = onnx.reference.ReferenceEvaluator(probe_model).run(None, feed)

        # Pass 2: rebuild with concrete, inferred output value_infos so the
        # importer sees correct result types (and check_model passes).
        output_infos = [
            onnx.helper.make_tensor_value_info(
                name,
                onnx.helper.np_dtype_to_tensor_dtype(arr.dtype),
                list(arr.shape),
            )
            for name, arr in zip(output_names, golden)
        ]
        return build_model(nodes, input_infos, output_infos, initializers=initializers)
