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
        pipeline_str = "torch-globalized-module-to-npcomp-backend-pipeline"
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
        pipeline_str = "torchscript-to-npcomp-backend-pipeline"
        if logging.debug_enabled():
            logging.debug(
                "Running Torch object graph lowering pipeline '{}'", pipeline_str)
        pm = PassManager.parse(pipeline_str)
        pm.run(imported_module)
    return imported_module
