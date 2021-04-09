#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

import torch

from mlir.ir import *
from mlir.passmanager import *
from npcomp.compiler.utils import logging

__all__ = [
    "lower_object_graph",
    "lower_module",
]

# The set of passes that lowers from a TorchScript object graph representation
# to a module semantics where symbols correspond to dotted paths into the
# module.
OBJECT_GRAPH_LOWERING_PASSES = (
    # Globalize the program. The rest of the compiler assumes a globalized
    # program, which makes all analyses and transforms significantly easier
    # to write.
    "torch-globalize-pipeline",
    # symbol-dce is currently needed for correctness, as we don't have a lowering
    # in the backend for torch.global_slot's.
    # Torch usually inserts a few unused global slots that are otherwise
    # bothersome because we don't currently have a lowering for them.
    # TODO: Support global slots in backends.
    "symbol-dce",
    # Incorporate user annotations and remove signature Python-isms.
    "torch-adjust-calling-conventions",
)

TORCH_TO_TCP_PASSES = (
    # Recognize ATen kernels.
    "func(aten-recognize-kernels)",

    # Convert the bulk of the program to ranked tensors with known dtype.
    # This is the input to the backend layer that we are aiming for.

    # First, unilaterally convert public functions to tensor.
    # The way this pass is currently written, this implies that
    # as pipeline authors, we are restricting our users to not be able to see
    # updates to "out params" on their public functions.
    # This is deemed ok for now.
    "numpy-public-functions-to-tensor",
    # Convert the bulk of non-ABI-visible arrays to tensors.
    "func(numpy-array-to-tensor)",
    # Do shape and dtype refinement.
    # We could do it sooner, but the pass currently doesn't have transfer
    # functions for array ops.
    "func(torch-refine-types)",
    # Propagate to ABI return types the shape/dtype information discovered by
    # the previous pass. Doing this is ABI-compatible for our backends.
    "numpy-refine-public-return",
    # Clean up a few stray array/tensor conversion remnants.
    "func(numpy-array-to-tensor)",

    # Lower to TCP (+ guards) which is the input to codegen backends.
    # Most of this should be subsumed by aten->linalg+guards conversions.
    # (the guard generation will be automated from the linalg Op DSL)
    "func(convert-aten-to-linalg)",
    "func(convert-aten-to-tcf)",
    "func(convert-tcf-to-std)",
    "func(convert-elementwise-to-linalg)",
)

def lower_module(imported_module: Module):
    """Compiles an imported module, with a flat list of functions.

    Args:
        imported_module: The MLIR module consisting of funcs and globals in
        the torch dialect. It is lowered in place.
    Returns:
        The imported_module, for convenience chaining methods.
    """
    with imported_module.context as context:
        if logging.debug_enabled():
            logging.debug("Initial PyTorch IR:\n{}", imported_module)
        # Frontend.
        pipeline_str = ",".join(TORCH_TO_TCP_PASSES)
        if logging.debug_enabled():
            logging.debug("Running Torch->TCP pipeline '{}'", pipeline_str)
        pm = PassManager.parse(pipeline_str)
        pm.run(imported_module)
        if logging.debug_enabled():
            logging.debug("TCP IR:\n{}", imported_module)
    return imported_module

def lower_object_graph(imported_module: Module):
    """Lowers an imported module that has TorchScript object graph semantics.

    Args:
        imported_module: The MLIR module consisting of IR as imported by the
        torch_mlir.import_module. It is lowered in place.
    Returns:
        The imported_module, for convenience chaining methods.
    """
    with imported_module.context as context:
        if logging.debug_enabled():
            logging.debug("Initial PyTorch object graph IR:\n{}", imported_module)

        # Object graph lowering.
        pipeline_str = ",".join(OBJECT_GRAPH_LOWERING_PASSES)
        if logging.debug_enabled():
            logging.debug(
                "Running Torch object graph lowering pipeline '{}'", pipeline_str)
        pm = PassManager.parse(pipeline_str)
        pm.run(imported_module)
    return lower_module(imported_module)
