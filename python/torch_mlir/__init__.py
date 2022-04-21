# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import List
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
    # This output type contains a mix of `linalg`-on-tensors ops, `scf`, and
    # `arith` ops (and also `math` and `tm_tensor`). It can be thought of
    # as taking the `TORCH` output type and lowering it so that tensor
    # computations are done with `linalg`-on-tensors ops.
    LINALG_ON_TENSORS = 2
    # This output type contains the raw imported `torch` dialect ops that are
    # obtained from the direct import of the JIT IR.
    # Since this bypasses the entire lowering pipeline, this is expected to
    # not be useful for most users, but can be useful for debugging.
    RAW = 3


def compile(model: torch.nn.Module,
            example_args: List[torch.Tensor],
            output_type: OutputType = OutputType.TORCH):
    """Convert a PyTorch model to MLIR.

    Args:
        model: The PyTorch model to convert.
        example_args: A list of example arguments to use when inferring the
            shapes of the arguments to `forward` method of the model.
            A single tensor is treated as a list of a single tensor.
        output_type: The kind of output to produce. See `OutputType` for more
            details.
    
    Returns:
        An MLIR module that contains the converted model in the specified
        output type.
    """

    # TODO: Don't hardcode "forward". See `torch.onnx.export` and
    # `torch.jit.trace_module` for API inspiration.
    # TODO: Support dynamic dimension sizes. See `torch.onnx.export`'s
    # `dynamic_axes` for API inspiration, or do something more ergonomic
    # like a tensor wrapper possibly.
    # TODO: Support tracing the model instead of scripting it.
    scripted = torch.jit.script(model)

    if isinstance(example_args, torch.Tensor):
        example_args = [example_args]

    class_annotator = ClassAnnotator()
    forward_annotation = [None]
    for arg in example_args:
        # Assume that all tensors have value semantics for now.
        forward_annotation.append((list(arg.shape), arg.dtype, True))
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
    else:
        assert output_type == OutputType.LINALG_ON_TENSORS
        run_pipeline_with_repro_report(
            mb.module,
            "torch-backend-to-linalg-on-tensors-backend-pipeline",
            "Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR")
    return mb.module
