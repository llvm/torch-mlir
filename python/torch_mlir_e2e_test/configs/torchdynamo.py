# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import List

import torch
import torch._dynamo as dynamo
import torch_mlir
import torch_mlir.dynamo
from torch_mlir.dynamo import make_simple_dynamo_backend
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

from torch_mlir_e2e_test.framework import TestConfig, Trace, TraceItem


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
    mlir_module = torch_mlir.compile(
        fx_graph, example_inputs, output_type="linalg-on-tensors")
    backend = refbackend.RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(mlir_module)
    loaded = backend.load(compiled)

    def compiled_callable(*inputs):
        inputs = [x.numpy() for x in inputs]
        result = loaded.forward(*inputs)
        if not isinstance(result, tuple):
            result = torch.from_numpy(result)
        else:
            result = tuple(torch.from_numpy(x) for x in result)
        return result
    return compiled_callable


class TorchDynamoTestConfig(TestConfig):
    """TestConfig that runs the torch.nn.Module with TorchDynamo"""

    def __init__(self):
        super().__init__()

    def compile(self, program: torch.nn.Module) -> torch.nn.Module:
        return program

    def run(self, artifact: torch.nn.Module, trace: Trace) -> Trace:
        # TODO: Deepcopy the torch.nn.Module, so that if the program is
        # stateful then it does not mutate the original compiled program.
        result: Trace = []
        for item in trace:
            f = lambda method, *inputs: method(*inputs)
            dynamo_f = dynamo.optimize(_refbackend_torchdynamo_backend)(f)
            output = dynamo_f(getattr(artifact, item.symbol), *item.inputs)
            result.append(
                TraceItem(symbol=item.symbol,
                          inputs=item.inputs,
                          output=output))
        return result
