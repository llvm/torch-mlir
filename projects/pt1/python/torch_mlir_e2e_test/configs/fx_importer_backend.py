# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import numpy as np
import torch
import torch.utils._pytree as pytree
from torch.export.graph_signature import OutputSpec, OutputKind
from torch.export import ExportedProgram

from torch_mlir import fx
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


class FxImporterTestConfig(TestConfig):
    """TestConfig that runs the torch.nn.Module with Fx Importer"""

    def __init__(self, backend, output_type="linalg-on-tensors"):
        super().__init__()
        self._backend = backend
        self._output_type = output_type

    def compile(
        self, program: torch.nn.Module, verbose: bool = False
    ) -> torch.nn.Module:
        return program

    def run(self, artifact: torch.nn.Module, trace: Trace) -> Trace:
        result: Trace = []
        for item in trace:
            prog: ExportedProgram = torch.export.export(artifact, tuple(item.inputs))
            module = fx.export_and_import(
                prog,
                output_type=self._output_type,
                func_name=artifact.__class__.__name__,
                # While the current e2e tests don't exercise symbolic shapes,
                # enabling this here ensures they don't regress either.
                import_symbolic_shape_expressions=True,
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
