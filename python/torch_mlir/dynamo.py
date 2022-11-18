# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import List

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from functorch._src.compile_utils import strip_overloads


import warnings
# https://github.com/pytorch/pytorch/issues/89064
warnings.filterwarnings("ignore", module="torch.jit._check")


def _unwrap_single_tuple_return(fx_g: torch.fx.GraphModule) -> bool:
    """Canonicalize single-element tuple returns to just the element.

    Returns:
        True if unwrapping took place, and false otherwise.
    """
    did_unwrap = False
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert len(
                node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, tuple):
                if len(node_arg) == 1:
                    node.args = (node_arg[0],)
                    did_unwrap = True
                    break

    if did_unwrap:
        fx_g.graph.lint()
        fx_g.recompile()
    return did_unwrap


def make_simple_dynamo_backend(user_backend):
    """Wrapper for functions intended to be used as TorchDynamo backends.

    This function simplifies a few of the steps that are required to make
    TorchDynamo work with Torch-MLIR.

    Args:
        user_backend: A function with the signature used by ordinary
            TorchDynamo backends. But the torch.fx.GraphModule passed to it
            will be normalized for consumption by `torch_mlir.compile`.
    Returns:
        A function with the signature used by TorchDynamo backends.
    """
    def wrapper_backend(fx_graph: torch.fx.GraphModule,
                        example_inputs: List[torch.Tensor]):
        did_unwrap = _unwrap_single_tuple_return(fx_graph)
        dispatcher_ops = make_fx(fx_graph)(*example_inputs)
        strip_overloads(dispatcher_ops)
        user_callable = user_backend(dispatcher_ops, example_inputs)

        def dynamo_callable(*inputs):
            result = user_callable(*inputs)
            return (result,) if did_unwrap else result
        return dynamo_callable
    return wrapper_backend
