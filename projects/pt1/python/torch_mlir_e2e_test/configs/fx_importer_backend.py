# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import Union, Optional, Sequence

import numpy as np
import torch
import torch.utils._pytree as pytree
from torch.export.graph_signature import OutputSpec, OutputKind
from torch.export import ExportedProgram

from torch_mlir import fx
from torch_mlir.compiler_utils import (
    run_pipeline_with_repro_report,
    lower_mlir_module,
    OutputType,
)
from torch_mlir.torchscript import (
    BACKEND_LEGAL_OPS,
    _canon_extra_library,
)
from torch_mlir_e2e_test.configs.utils import (
    recursively_convert_to_numpy,
    recursively_convert_from_numpy,
)
from torch_mlir_e2e_test.framework import TestConfig, Trace, TraceItem


def refine_result_type(_result):
    if isinstance(_result, tuple):
        return tuple(refine_result_type(x) for x in _result)
    elif isinstance(_result, np.ndarray):
        return torch.from_numpy(_result)
    elif isinstance(_result, (bool, int, float)):
        return _result
    else:
        raise ValueError(f"Unhandled return type {type(_result)}")


def jit(
    prog: ExportedProgram,
    func_name: str,
    output_type: Union[str, "OutputType"] = OutputType.TORCH,
    backend_legal_ops: Optional[Sequence[str]] = None,
    extra_library=None,
    verbose: bool = False,
):
    if extra_library is None:
        extra_library = []
    mlir_module = None

    extra_library_file_name = _canon_extra_library(extra_library)
    output_type = OutputType.get(output_type)
    if backend_legal_ops is not None:
        if output_type != OutputType.TORCH:
            raise Exception(
                "`backend_legal_ops` is only valid with the " "`torch` output type"
            )
        backend_legal_ops = list(sorted(set(backend_legal_ops)))
    else:
        backend_legal_ops = BACKEND_LEGAL_OPS.get(output_type, [])

    option_string = (
        "{backend-legal-ops="
        + ",".join(backend_legal_ops)
        + " extra-library="
        + extra_library_file_name
        + "}"
    )

    mlir_module = fx.export_and_import(prog, func_name=func_name)
    assert mlir_module is not None
    run_pipeline_with_repro_report(
        mlir_module,
        f"builtin.module(torch-simplification-pipeline)",
        "Simplification pipeline for torch dialect",
    )
    run_pipeline_with_repro_report(
        mlir_module,
        f"builtin.module(torch-function-to-torch-backend-pipeline{option_string})",
        "Lowering TorchFX IR -> Torch Backend IR",
    )

    return lower_mlir_module(verbose, output_type, mlir_module)


class FxImporterTestConfig(TestConfig):
    """TestConfig that runs the torch.nn.Module with Fx Importer"""

    def __init__(self, backend, output_type="linalg-on-tensors"):
        super().__init__()
        self._backend = backend
        self._output_type = output_type

    def compile(self, program: torch.nn.Module) -> torch.nn.Module:
        return program

    def run(self, artifact: torch.nn.Module, trace: Trace) -> Trace:
        result: Trace = []
        for item in trace:
            prog = torch.export.export(artifact, tuple(item.inputs))
            module = jit(
                prog,
                func_name=artifact.__class__.__name__,
                output_type=self._output_type,
            )
            module = self._backend.compile(module)
            backend_module = self._backend.load(module)
            params = {
                # **dict(artifact.named_parameters(remove_duplicate=False)),
                **dict(artifact.named_buffers(remove_duplicate=False)),
            }
            params_flat, params_spec = pytree.tree_flatten(params)
            params_flat = list(params_flat)
            with torch.no_grad():
                numpy_inputs = recursively_convert_to_numpy(params_flat + item.inputs)
            outputs = getattr(backend_module, artifact.__class__.__name__)(
                *numpy_inputs
            )
            output = refine_result_type(outputs)
            if isinstance(output, (tuple, list)):
                user_output = []
                out_spec: OutputSpec
                for val, out_spec in zip(output, prog.graph_signature.output_specs):
                    if out_spec.kind == OutputKind.USER_OUTPUT:
                        user_output.append(val)
                output = tuple(user_output)
            result.append(
                TraceItem(symbol=item.symbol, inputs=item.inputs, output=output)
            )
        return result
