# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import List, Union, Optional, Sequence

import numpy as np
import torch
import torch.utils._pytree as pytree
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import (
    make_boxed_compiler,
    get_aot_compilation_context,
    set_model_name,
)

from torch_mlir._dynamo_fx_importer import import_fx_graph_as_func
from torch_mlir.dynamo import _get_decomposition_table
from torch_mlir.torchscript import (
    _example_args,
    OutputType,
    BACKEND_LEGAL_OPS,
    run_pipeline_with_repro_report,
    _lower_mlir_module,
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


def _returns_empty_tuple(fx_graph: torch.fx.GraphModule) -> bool:
    for node in fx_graph.graph.nodes:
        if node.op == "output":
            assert len(
                node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if node_arg != ():
                return False
    return True


def jit(
    model: torch.nn.Module,
    example_args: _example_args,
    output_type: Union[str, "OutputType"] = OutputType.TORCH,
    backend_legal_ops: Optional[Sequence[str]] = None,
    extra_library=None,
    verbose: bool = False,
):
    if extra_library is None:
        extra_library = []
    import torch._dynamo as dynamo

    mlir_module = None

    extra_library_file_name = _canon_extra_library(extra_library)
    output_type = OutputType.get(output_type)
    if backend_legal_ops is not None:
        if output_type != OutputType.TORCH:
            raise Exception("`backend_legal_ops` is only valid with the "
                            "`torch` output type")
        backend_legal_ops = list(sorted(set(backend_legal_ops)))
    else:
        backend_legal_ops = BACKEND_LEGAL_OPS.get(output_type, [])

    @make_boxed_compiler
    def my_aot_autograd_backend(gm: torch.fx.GraphModule,
                                _example_inputs: List[torch.Tensor]):
        # Torch-MLIR does not support returning an empty tuple. The reason is
        # that both returning an empty tuple and returning `None` results in MLIR
        # functions that have as a return type `()`. In other words, there is no
        # way of differentiating between the two.
        assert not _returns_empty_tuple(gm), "encountered graph that does not return anything"

        nonlocal mlir_module
        *_, model_name, nth_graph = get_aot_compilation_context()
        mlir_module = import_fx_graph_as_func(gm.graph, model_name)
        return gm

    my_backend = aot_autograd(fw_compiler=my_aot_autograd_backend,
                              decompositions=_get_decomposition_table)

    with torch.no_grad():
        set_model_name(model.__class__.__name__)
        torch._dynamo.reset()
        dynamo_f = dynamo.optimize(my_backend, nopython=True)(
            lambda method, *inputs: method(*inputs))
        dynamo_f(lambda *inputs: model(*[x.clone() for x in inputs]),
                 *example_args)
        option_string = ("{backend-legal-ops=" + ",".join(backend_legal_ops) +
                         " extra-library=" + extra_library_file_name + "}")
        assert mlir_module is not None
        run_pipeline_with_repro_report(
            mlir_module,
            # f"builtin.module(torch-function-to-torch-backend-pipeline{option_string})",
            f"builtin.module(torch-lower-to-backend-contract)",
            "Lowering TorchFX IR -> Torch Backend IR",
        )

    return _lower_mlir_module(verbose, output_type, mlir_module)


class TorchDynamoTestConfig(TestConfig):
    """TestConfig that runs the torch.nn.Module with TorchDynamo"""

    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def compile(self, program: torch.nn.Module) -> torch.nn.Module:
        return program

    def run(self, artifact: torch.nn.Module, trace: Trace) -> Trace:
        result: Trace = []
        for item in trace:
            module = jit(artifact,
                         item.inputs,
                         output_type="linalg-on-tensors")
            module = self.backend.compile(module)
            backend_module = self.backend.load(module)
            params = {
                **dict(artifact.named_parameters(remove_duplicate=False)),
                **dict(artifact.named_buffers(remove_duplicate=False)),
            }
            params_flat, params_spec = pytree.tree_flatten(params)
            params_flat = list(params_flat)
            with torch.no_grad():
                numpy_inputs = recursively_convert_to_numpy(params_flat +
                                                            item.inputs)
            outputs = getattr(backend_module,
                              artifact.__class__.__name__)(*numpy_inputs)
            output = refine_result_type(outputs)
            result.append(
                TraceItem(symbol=item.symbol,
                          inputs=item.inputs,
                          output=output))
        return result
