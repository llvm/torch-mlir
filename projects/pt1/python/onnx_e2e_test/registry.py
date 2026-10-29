# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import Callable, List

import onnx

from torch_mlir.compiler_utils import TensorPlaceholder

from .framework import OnnxTest

# The global registry of ONNX end-to-end tests.
GLOBAL_ONNX_TEST_REGISTRY: List[OnnxTest] = []
# Ensure that there are no duplicate names in the global registry.
_SEEN_UNIQUE_NAMES: set = set()


def register_onnx_test(cls):
    """Class-based registration for ONNX end-to-end test cases.

    Decorate an ``OnnxTestCase`` subclass; its name is the ``unique_name``, its
    ``build`` classmethod is the model factory, and its input placeholders are
    derived from the ``@annotate_inputs`` on ``graph``.  See ``OnnxTestCase``.
    """
    _register(cls.__name__, cls.build, cls.input_placeholders())
    return cls


def _register(
    unique_name: str,
    model_factory: Callable[[], onnx.ModelProto],
    inputs: List[TensorPlaceholder],
):
    if unique_name in _SEEN_UNIQUE_NAMES:
        raise Exception(
            f"Duplicate test name: '{unique_name}'. Please make sure that "
            "each registered ONNX test has a unique name."
        )
    _SEEN_UNIQUE_NAMES.add(unique_name)
    GLOBAL_ONNX_TEST_REGISTRY.append(
        OnnxTest(
            unique_name=unique_name,
            model_factory=model_factory,
            inputs=inputs,
        )
    )
