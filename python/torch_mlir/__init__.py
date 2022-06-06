# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import Sequence, Union, List
from enum import Enum

import torch

from torch_mlir.passmanager import PassManager
from .compiler_utils import run_pipeline_with_repro_report
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder


class OutputType(Enum):
    """The kind of output that `torch_mlir.compile` can produce.

    In MLIR terminology, this describes the mix of dialects that will be
    produced by the conversion process.
    """
    # This output type consists of `torch` dialect ops that have been converted
    # maximally to value semantics, decomposed, and shapes have been inferred.
    TORCH = 0
    # This output type consists of `tosa` dialect ops. It can be thought of
    # as taking the `TORCH` output type and lowering it to TOSA.
    TOSA = 1
    # The output type contains a mix of `linalg`-on-tensors ops, `scf`, and
    # `arith` ops (and also `math` and `tm_tensor`). It can be thought of
    # as taking the `TORCH` output type and lowering it so that tensor
    # computations are done with `linalg`-on-tensors ops.
    LINALG_ON_TENSORS = 2
    # Raw output of the JIT IR importer. This is not expected to be useful
    # for end-users, but can be convenient for development or reporting bugs.
    RAW = 3


class TensorPlaceholder:
    """A class that represents a formal parameter of a given shape and dtype.

    This class can be constructed explicitly from a shape and dtype:
    ```python
    placeholder = TensorPlaceholder([3, 4], torch.float32)
    ```

    This class can also be constructed from a `torch.Tensor` which is already
    known to be a valid input to the function. In this case, a set of
    dynamic axes are allowed to be specified.
    ```python
    placeholder = TensorPlaceholder.like(torch.ones(3, 4), dynamic_axes=[1])
    # Equivalent to `TensorPlaceholder([3, -1], torch.float32)`
    ```
    """

    def __init__(self, shape: List[int], dtype: torch.dtype):
        """Create a tensor with shape `shape` and dtype `dtype`.

        Args:
            shape: The shape of the tensor. A size of `-1` indicates that the
            dimension has an unknown size.
            dtype: The dtype of the tensor.
        """
        self.shape = shape
        self.dtype = dtype

    @staticmethod
    def like(tensor: torch.Tensor, dynamic_axes: List[int] = None):
        """Create a tensor placeholder that is like the given tensor.

        Args:
            tensor: The tensor to create a placeholder for.
            dynamic_axes: A list of dynamic axes. If specified, the compiled
            module will allow those axes to be any size at runtime.
        """
        if dynamic_axes is None:
            dynamic_axes = []
        shape = []
        for i, dim in enumerate(tensor.shape):
            if i in dynamic_axes:
                shape.append(-1)
            else:
                shape.append(dim)
        return TensorPlaceholder(shape, tensor.dtype)


_example_arg = Union[TensorPlaceholder, torch.Tensor]


def compile(model: torch.nn.Module,
            example_args: Union[_example_arg, Sequence[_example_arg]],
            output_type: OutputType = OutputType.TORCH,
            use_tracing=False):
    """Convert a PyTorch model to MLIR.

    Args:
        model: The PyTorch model to convert.
        example_args: A list of example arguments to use when inferring the
            shapes of the arguments to `forward` method of the model.
            A single tensor is treated as a list of a single tensor.
            A TensorPlaceholder object is also allowed in the place of any
            Tensor.
        output_type: The kind of output to produce. See `OutputType` for more
            details.
        use_tracing: If True, use `torch.jit.trace` to convert the model to
            JIT IR rather than `torch.jit.script`.

    Returns:
        An MLIR module that contains the converted model in the specified
        output type.
    """

    # Special case -- many models have just one input, so canonicalize a single
    # tensor to a list of a single tensor to make the API more ergonomic.
    if isinstance(example_args, (torch.Tensor, TensorPlaceholder)):
        example_args = (example_args,)

    # TODO: Don't hardcode "forward". See `torch.onnx.export` and
    # `torch.jit.trace_module` for API inspiration.
    if use_tracing:
        scripted = torch.jit.trace(model, tuple(example_args))
    else:
        scripted = torch.jit.script(model)

    # Convert all concrete inputs to TensorPlaceholder's, for consistency.
    arg_placeholders = []
    for arg in example_args:
        if isinstance(arg, TensorPlaceholder):
            arg_placeholders.append(arg)
        else:
            assert isinstance(arg, torch.Tensor)
            arg_placeholders.append(TensorPlaceholder.like(arg))

    class_annotator = ClassAnnotator()
    forward_annotation = [None]
    for arg in arg_placeholders:
        # Assume that all tensors have value semantics for now.
        forward_annotation.append((arg.shape, arg.dtype, True))
    class_annotator.exportNone(scripted._c._type())
    class_annotator.exportPath(scripted._c._type(), ["forward"])
    class_annotator.annotateArgs(
        scripted._c._type(), ["forward"], forward_annotation)

    mb = ModuleBuilder()
    mb.import_module(scripted._c, class_annotator)

    if output_type == OutputType.RAW:
        return mb.module

    run_pipeline_with_repro_report(mb.module,
                                   "torchscript-module-to-torch-backend-pipeline",
                                   "Lowering TorchScript IR -> Torch Backend IR")

    if output_type == OutputType.TORCH:
        return mb.module

    if output_type == OutputType.TOSA:
        run_pipeline_with_repro_report(
            mb.module,
            "torch-backend-to-tosa-backend-pipeline",
            "Lowering Torch Backend IR -> TOSA Backend IR")
        return mb.module

    if output_type == OutputType.LINALG_ON_TENSORS:
        run_pipeline_with_repro_report(
            mb.module,
            "torch-backend-to-linalg-on-tensors-backend-pipeline",
            "Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR")
        return mb.module

    raise Exception(f"Unknown OutputType: {output_type}")
