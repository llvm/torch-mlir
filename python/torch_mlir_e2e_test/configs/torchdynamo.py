# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import List

import numpy
import torch
import torch._dynamo as dynamo
import torch_mlir
import torch_mlir.dynamo
from torch_mlir.dynamo import make_simple_dynamo_backend
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

from torch_mlir_e2e_test.framework import TestConfig, Trace, TraceItem

def _returns_empty_tuple(fx_graph: torch.fx.GraphModule) -> bool:
    for node in fx_graph.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if node_arg != ():
                return False
    return True

@make_simple_dynamo_backend
def _refbackend_torchdynamo_backend(fx_graph: torch.fx.GraphModule,
                                    example_inputs: List[torch.Tensor]):
    # Use the LinalgOnTensors backend, since it is the most complete.
    # In theory we could mix and match TorchDynamo with the other backends,
    # since they all lower through the same backend contract.
    # For now, testing-wise, it doesn't make sense to test those configurations.
    # We really just want to check the TorchDynamo frontend.
    #
    # Longer-term we will need to do something more sophisticated here.
    # As per the long-term roadmap:
    # https://github.com/llvm/torch-mlir/blob/main/docs/long_term_roadmap.md#refactoring-the-frontend
    # We will eventually have a configuration that uses new PyTorch infra and
    # skips the entire "frontend" part. We currently don't have any code
    # for that right now since it is still very early stages, but eventually
    # this Config should test that path (and maybe the current behavior can
    # be moved to a `legacy_frontend_via_torchdynamo` config).

    # Torch-MLIR does not support returning an empty tuple. The reason is
    # that both returning an empty tuple and returning `None` results in MLIR
    # functions that have as a return type `()`. In other words, there is no
    # way of differentiating between the two. Moreover, since Torch-MLIR treats
    # inputs as having value semantics, graphs that return nothing are no-ops to
    # Torch-MLIR.
    if _returns_empty_tuple(fx_graph):
        return fx_graph

    mlir_module = torch_mlir.compile(
        fx_graph, example_inputs, output_type="linalg-on-tensors")
    backend = refbackend.RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(mlir_module)
    loaded = backend.load(compiled)

    def compiled_callable(*inputs):
        def refine_result_type(_result):
            if isinstance(_result, tuple):
                return tuple(refine_result_type(x) for x in _result)
            elif isinstance(_result, numpy.ndarray):
                return torch.from_numpy(_result)
            elif isinstance(_result, (bool, int, float)):
                return _result
            else:
                raise ValueError(f"Unhandled return type {type(_result)}")
        inputs = [x.numpy() for x in inputs]
        result = loaded.forward(*inputs)
        return refine_result_type(result)
    return compiled_callable


class TorchDynamoTestConfig(TestConfig):
    """TestConfig that runs the torch.nn.Module with TorchDynamo"""

    def __init__(self):
        super().__init__()

    def compile(self, program: torch.nn.Module) -> torch.nn.Module:
        return program

    def run(self, artifact: torch.nn.Module, trace: Trace) -> Trace:
        def item_symbol_that_clones_inputs(*inputs):
            cloned_inputs = [x.clone() for x in inputs]
            result = getattr(artifact, item.symbol)(*cloned_inputs)
            return result
        # TODO: Deepcopy the torch.nn.Module, so that if the program is
        # stateful then it does not mutate the original compiled program.
        result: Trace = []
        for item in trace:
            f = lambda method, *inputs: method(*inputs)
            dynamo_f = dynamo.optimize(_refbackend_torchdynamo_backend)(f)
            output = dynamo_f(item_symbol_that_clones_inputs, *item.inputs)
            result.append(
                TraceItem(symbol=item.symbol,
                          inputs=item.inputs,
                          output=output))
        return result
