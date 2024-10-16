# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import numpy as np
import torch
import torch.utils._pytree as pytree
from torch.export.graph_signature import OutputSpec, OutputKind
from torch.export import ExportedProgram
from torch._dynamo.backends.common import aot_autograd

from torch_mlir import fx
from torch_mlir_e2e_test.configs.utils import (
    recursively_convert_to_numpy,
    recursively_convert_from_numpy,
)
from torch_mlir_e2e_test.framework import TestConfig, Trace, TraceItem
from torch_mlir_e2e_test.annotations import TORCH_MLIR_ARG_ANNOTATIONS_ATTR_NAME


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

    def __init__(self, backend, output_type="linalg-on-tensors", torch_compile=False):
        super().__init__()
        self._backend = backend
        self._torch_compile = torch_compile
        self._output_type = output_type

    def compile(
        self, program: torch.nn.Module, verbose: bool = False
    ) -> torch.nn.Module:
        return program

    def run(self, artifact: torch.nn.Module, trace: Trace):
        return (
            self._export_run(artifact, trace)
            if not self._torch_compile
            else self._stateless_run(artifact, trace)
        )

    def _stateless_run(self, artifact: torch.nn.Module, trace: Trace):
        dynamic_argument_pos = None
        dynamic_dim_pos = None
        annotations = getattr(artifact.forward, TORCH_MLIR_ARG_ANNOTATIONS_ATTR_NAME)
        for i, annotation in enumerate(annotations):
            if i == 0:  # Skip the "self" annotation.
                continue
            if not annotation[2]:
                raise ValueError(
                    "Can only compile inputs annotated as having value semantics."
                )
            for dim_i, dim in enumerate(annotation[0]):
                if dim == -1:
                    dynamic_argument_pos = i - 1
                    dynamic_dim_pos = dim_i
                    break
            if dynamic_argument_pos is not None:
                break
        result: Trace = []
        for item in trace:

            def _base_backend(gm: torch.fx.GraphModule, example_inputs):
                for node in gm.graph.nodes:
                    if node.op == "placeholder":
                        if (
                            isinstance(node.meta["val"], torch.SymInt)
                            and not node.users
                        ):
                            gm.graph.erase_node(node)
                module = fx.stateless_fx_import(
                    gm,
                    output_type=self._output_type,
                    model_name=artifact.__class__.__name__,
                )
                module = self._backend.compile(module)
                backend_module = self._backend.load(module)

                def invoke_func(*torch_inputs):
                    torch_inputs = [
                        x
                        for x in filter(
                            lambda i: isinstance(i, torch.Tensor), torch_inputs
                        )
                    ]
                    with torch.no_grad():
                        numpy_inputs = recursively_convert_to_numpy(torch_inputs)
                    return recursively_convert_from_numpy(
                        getattr(backend_module, artifact.__class__.__name__)(
                            *numpy_inputs
                        )
                    )

                return invoke_func

            fw_compiler = aot_autograd(fw_compiler=_base_backend)
            if dynamic_argument_pos is not None:
                torch._dynamo.mark_dynamic(
                    item.inputs[dynamic_argument_pos], dynamic_dim_pos
                )
            module = torch.compile(artifact, backend=fw_compiler)
            outputs = module(*item.inputs)
            result.append(
                TraceItem(symbol=item.symbol, inputs=item.inputs, output=outputs)
            )
        return result

    def _export_run(self, artifact: torch.nn.Module, trace: Trace) -> Trace:
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
