# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import List, Optional, Tuple

import torch

import torch_mlir
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator

# Decorators

# Currently, these decorators are very low-level and map 1:1 with
# methods on `ClassAnnotator`. Eventually, we expect there to
# be a more elaborate Python layer which allows all the different annotations
# to be expressed conveniently and gives clearer error reports when
# the annotations aren't acceptable.

# This module is kept separate from torch_mlir_e2e_test.annotations so that
# we can use that module from code without C++ dependencies, which prevent us
# from interfacing the test framework across environments.

# Utilities for extracting decorated information into ClassAnnotator.

def _recursively_extract_annotations(
        module: torch.nn.Module, scripted: torch.jit.ScriptModule,
        class_annotator: ClassAnnotator):
    assert module.__class__.__name__ == scripted.original_name or (
        isinstance(module, torch.jit.RecursiveScriptModule) and module is
        scripted), "script module does not come from specified module"

    # Extract information on methods.
    for method_name, scripted_method in scripted.__dict__.items():
        if not isinstance(scripted_method, torch.ScriptMethod):
            continue
        method = getattr(module, method_name)
        if hasattr(method, '_torch_mlir_export'):
            class_annotator.exportPath(scripted._c._type(), [method_name])
        if hasattr(method, '_torch_mlir_arg_annotations'):
            class_annotator.annotateArgs(
                scripted._c._type(), [method_name],
                method._torch_mlir_arg_annotations)
    # Recurse.
    for name, child in module.named_children():
        scripted_child = getattr(scripted, name)
        _recursively_extract_annotations(child, scripted_child,
                                         class_annotator)


def extract_annotations(program: torch.nn.Module,
                        scripted: torch.jit.ScriptModule,
                        class_annotator: ClassAnnotator):
    """Populate the ClassAnnotator with annotations extracted from `program`."""
    class_annotator.exportNone(scripted._c._type())
    _recursively_extract_annotations(program, scripted, class_annotator)
