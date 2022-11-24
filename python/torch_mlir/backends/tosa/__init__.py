# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import Optional, Sequence, Union

import torch
import torch_mlir

from ...compiler_utils import run_pipeline_with_repro_report

def compile(model: torch.nn.Module,
            example_args: Union[torch_mlir._example_arg, Sequence[torch_mlir._example_arg]],
            use_tracing: bool = False,
            ignore_traced_shapes = False,
            backend_legal_ops: Optional[Sequence[str]] = None,
            backend_custom_ops: Optional[Sequence[str]] = None,
            verbose: bool = False):
    """Convert a PyTorch model to TOSA IR in MLIR.

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
        ignore_traced_shapes: If True, ignore the shapes that were observed
            during tracing. This should only be used if one knows that the
            original traced program would result in the same trace (modulo
            shapes) for all shape combinations implied by any
            `TensorPlaceholder`'s used as `example_args`. Also,
            strictly-speaking, this option covers dtypes too, but we just say
            "shapes" to be succinct.
        backend_legal_ops: A list of ops that should be considered legal for
            the backend. An op that is considered legal will not be decomposed.
            This option is only valid with the `"torch"` output type.
        backend_custom_ops: A list of ops to be converted to the custom ops in
            the backend dialect.
        verbose: If true, print extra information about the conversion.

    Returns:
        An MLIR module that contains the converted model in TOSA IR.
    """

    if backend_legal_ops is None:
        backend_legal_ops = torch_mlir.BACKEND_LEGAL_OPS.get(torch_mlir.OutputType.TOSA, [])

    module = torch_mlir.compile(model, example_args, torch_mlir.OutputType.TORCH, use_tracing,
                                ignore_traced_shapes, backend_legal_ops, verbose)
    
    if backend_custom_ops is None:
        backend_custom_ops = []

    backend_option_string = "{custom-ops=" + ",".join(backend_custom_ops) + "}"
    run_pipeline_with_repro_report(
        module,
        f"builtin.module(torch-backend-to-tosa-backend-pipeline{backend_option_string})",
        "Lowering Torch Backend IR -> TOSA Backend IR")
    if verbose:
        print("\n====================")
        print("TOSA Backend IR")
        print(module)
    return module

