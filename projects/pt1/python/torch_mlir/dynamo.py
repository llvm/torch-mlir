# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import List
from ._version import torch_version_for_comparison, version

import torch
from torch._functorch.compile_utils import strip_overloads
from torch._decomp import get_decompositions
from torch._dynamo.backends.common import aot_autograd
import functorch

import warnings
# https://github.com/pytorch/pytorch/issues/89064
warnings.filterwarnings("ignore", module="torch.jit._check")


def _get_decomposition_table():
    """Get a decomposition table suitable for Torch-MLIR.

    Sometimes TorchDynamo traces slightly different ops than what TorchScript
    captures. Historically we have been driven by the ops captured by
    TorchScript, so we try to decompose the ops captured by TorchDynamo into
    other ops that we already support.

    There isn't a highly principled solution here. Torch-MLIR currently supports
    a somewhat random set of ops, added in a demand-driven way over time,
    including direct backend support and decompositions internal to Torch-MLIR.
    As described in the
    [long-term roadmap](https://github.com/llvm/torch-mlir/blob/main/docs/long_term_roadmap.md),
    eventually this situation is expected to be made a lot more principled
    by aligning more with how Torch-MLIR would have looked if some of the new
    upstream PyTorch infra had been available at the beginning -- in particular
    the new decomposition infra and PrimTorch.
    """
    aten = torch.ops.aten
    decomp_list = [
        aten._adaptive_avg_pool2d,
        aten.std.correction,
        aten.dot,
        # TODO: Backends probably want to support this directly without
        # decomposition.
        # Our current situation with batch norm is a bit of a mess.
        # aten.batch_norm has direct backend lowerings,
        # aten.native_batch_norm gets decomposed into elementwise/reductions
        # by DecomposeComplexOps (no backend marks it as backend-legal).
        # Neither appears to support the "training" mode
        # (the upstream decomposition we use here does), even though we have
        # support for aten.native_batch_norm_backward.
        aten._native_batch_norm_legit_functional,
        aten.native_group_norm,
        aten.split.Tensor,
        aten.split_with_sizes,
        aten.norm.ScalarOpt_dim,
        aten.embedding_dense_backward,
        aten.native_layer_norm_backward,
        aten.slice_backward,
        aten.select_backward,
        aten.upsample_bilinear2d.vec,
        aten.mse_loss_backward,
        aten.native_group_norm_backward,
        aten.sigmoid_backward,
        aten._native_batch_norm_legit,
        aten.squeeze,
    ]
    # TODO: enable test once 2.1.0 is stable
    if torch_version_for_comparison() >= version.parse("2.1.0.dev"):
        decomp_list += [aten._native_batch_norm_legit_no_training]
    return get_decompositions(decomp_list)


def _adjust_calling_convention(gm: torch.fx.GraphModule) -> bool:
    """Canonicalize the calling convention to the one that Torch-MLIR supports.

    The MLIR codebase currently supports importing functions that have either
    a None return value, a single return value or a non-singleton tuple of
    return values. But various situations create functions with single-element
    tuples, or lists instead of tuples. This function adjusts the calling
    conventions to match, and returns the information needed for the calling
    code to reconstruct the original calling convention.

    Returns:
        Two booleans
        - The first indicates if a single-element tuple/list return
          was converted to a return of the element itself.
        - The second indicates if a list return was converted to a tuple.
    """
    did_unwrap_single_element = False
    did_convert_list_to_tuple = False
    for node in gm.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, \
                "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, tuple):
                if len(node_arg) == 1:
                    node.args = (node_arg[0],)
                    did_unwrap_single_element = True
                    break
            if isinstance(node_arg, list):
                if len(node_arg) == 1:
                    node.args = (node_arg[0],)
                    did_unwrap_single_element = True
                    did_convert_list_to_tuple = True
                    break
                else:
                    node.args= (tuple(node_arg),)
                    did_convert_list_to_tuple = True
                    break

    if did_unwrap_single_element:
        gm.graph.lint()
        gm.recompile()
    return did_unwrap_single_element, did_convert_list_to_tuple


def make_simple_dynamo_backend(user_backend):
    """Wrapper for functions intended to be used as TorchDynamo backends.

    This function simplifies a few of the steps that are required to make
    TorchDynamo work with Torch-MLIR.

    Args:
        user_backend: A function with the signature used by ordinary
            TorchDynamo backends. But the torch.fx.GraphModule passed to it
            will be normalized for consumption by `torchscript.compile`.
    Returns:
        A function with the signature used by TorchDynamo backends.
    """
    def wrapper_backend(gm: torch.fx.GraphModule,
                        example_inputs: List[torch.Tensor]):
        did_unwrap_single_element, did_convert_list_to_tuple = \
            _adjust_calling_convention(gm)
        strip_overloads(gm)
        user_callable = user_backend(gm, example_inputs)

        # TODO: Have a consistent story about the boxed calling convention.
        # (for more details on this remove this decorator and look at the warning)
        # See https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670 for rationale.
        @functorch.compile.make_boxed_func
        def dynamo_callable(*inputs):
            result = user_callable(*inputs)
            if did_unwrap_single_element:
                result = (result,)
            if did_convert_list_to_tuple:
                result = list(result)
            return result
        return dynamo_callable
    return aot_autograd(fw_compiler=wrapper_backend,
                        decompositions=_get_decomposition_table)
