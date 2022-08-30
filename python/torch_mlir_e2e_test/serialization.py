# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""
# Serialization utilities for the end-to-end test framework.

It is sometimes useful to be able to serialize tests to disk, such as when
multiple tests require mutally incompatible PyTorch versions. This takes
advantage of the strong backwards compatibility of serialized TorchScript, which
generally allows all such programs to be loaded to an appropriately recent
PyTorch.
"""

from typing import List, Tuple, Optional, NamedTuple

import io
import os
import pickle

import torch

from .framework import generate_golden_trace, Test, Trace
from .annotations import ArgAnnotation, export, annotate_args, TORCH_MLIR_EXPORT_ATTR_NAME, TORCH_MLIR_ARG_ANNOTATIONS_ATTR_NAME
from .registry import GLOBAL_TEST_REGISTRY

# ==============================================================================
# Annotation serialization
# ==============================================================================


class SerializableMethodAnnotation(NamedTuple):
    method_name: str
    export: Optional[bool]
    arg_annotations: Optional[List[ArgAnnotation]]


class SerializableModuleAnnotations(NamedTuple):
    method_annotations: List[SerializableMethodAnnotation]
    submodule_annotations: List[Tuple[str, "SerializableModuleAnnotations"]]


def extract_serializable_annotations(
        module: torch.nn.Module) -> SerializableModuleAnnotations:
    module_annotations = SerializableModuleAnnotations(
        method_annotations=[], submodule_annotations=[])
    # Extract information on methods.
    for method_name, method in module.__dict__.items():
        # See if it is a method.
        if not callable(method):
            continue
        export = None
        arg_annotations = None
        if hasattr(method, TORCH_MLIR_EXPORT_ATTR_NAME):
            export = method._torch_mlir_export
        if hasattr(method, TORCH_MLIR_ARG_ANNOTATIONS_ATTR_NAME):
            arg_annotations = method._torch_mlir_arg_annotations
        if export is not None and arg_annotations is not None:
            module_annotations.method_annotations.append(
                SerializableMethodAnnotation(method_name=method_name,
                                             export=export,
                                             arg_annotations=arg_annotations))

    # Recurse.
    for name, child in module.named_children():
        annotations = extract_serializable_annotations(child)
        module_annotations.submodule_annotations.append((name, annotations))
    return module_annotations


def apply_serializable_annotations(module: torch.nn.Module,
                                   annotations: SerializableModuleAnnotations):
    # Apply annotations to methods.
    for method_annotation in annotations.method_annotations:
        # Imitate use of the decorators to keep a source of truth there.
        if method_annotation.export is not None:
            setattr(module, method_annotation.method_name,
                    export(getattr(module, method_annotation.method_name)))
        if method_annotation.arg_annotations is not None:
            setattr(
                module, method_annotation.method_name,
                annotate_args(method_annotation.arg_annotations)(getattr(
                    module, method_annotation.method_name)))

    # Recurse.
    for name, submodule_annotations in annotations.submodule_annotations:
        child = getattr(module, name)
        apply_serializable_annotations(child, submodule_annotations)

# ==============================================================================
# Serializable test definition
# ==============================================================================


class SerializableTest(NamedTuple):
    """A self-contained representation of a test that can be pickled.

    We use serialized TorchScript programs here for two reasons:
    1. The PyTorch pickling story isn't great, so in order to reliably pickle
       this class, we rely on having the serialized bytes for the TorchScript
       module already given to us.
    2. The choice of a TorchScript module vs `torch.nn.Module` boils down to
       the fact that `torch.nn.Module` cannot be deserialized without pulling
       in the same set of Python dependencies that were used to serialize it
       in the first place. This would defeat one of the
       main use cases of this class, which is to transport a test from an
       environment with a set of heavy dependencies to a dependency-light one.
       Since TorchScript modules are self-contained, they fit the bill
       perfectly.
    """
    # See unique_name on `Test`.
    unique_name: str
    # Serialized TorchScript program.
    program: bytes
    # Trace for execution testing.
    trace: Trace

    def as_test(self) -> Test:
        """Create a `Test` from this class."""
        # Conform the serialized program to the interface expected by Test.
        # This is a bit of a hack, but it's the only way to keep the layering
        # straight.
        def factory():
            _extra_files = {"annotations.pkl": ""}
            module = torch.jit.load(io.BytesIO(self.program),
                                    _extra_files=_extra_files)
            # Load the pickled annotations.
            annotations = pickle.loads(_extra_files["annotations.pkl"])
            apply_serializable_annotations(module, annotations)
            return module

        def invoker(module, tu):
            for item in self.trace:
                attr = module
                for part in item.symbol.split("."):
                    attr = getattr(attr, part)
                attr(*item.inputs)

        return Test(
            unique_name=self.unique_name,
            program_factory=factory,
            program_invoker=invoker,
        )


# ==============================================================================
# Filesystem operations
# ==============================================================================

def serialize_all_tests_to(output_dir: str):
    serializable_tests = []
    for test in GLOBAL_TEST_REGISTRY:
        trace = generate_golden_trace(test)
        module = torch.jit.script(test.program_factory())
        torchscript_module_bytes = module.save_to_buffer({
            "annotations.pkl":
            pickle.dumps(extract_serializable_annotations(module))
        })
        serializable_tests.append(
            SerializableTest(unique_name=test.unique_name,
                             program=torchscript_module_bytes,
                             trace=trace))
    for test in serializable_tests:
        with open(os.path.join(output_dir, f"{test.unique_name}.pkl"),
                  "wb") as f:
            pickle.dump(test, f)


def deserialize_all_tests_from(serialized_test_dir: str):
    for root, _, files in os.walk(serialized_test_dir):
        for filename in files:
            with open(os.path.join(root, filename), 'rb') as f:
                GLOBAL_TEST_REGISTRY.append(pickle.load(f).as_test())
