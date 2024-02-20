# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import abc
from typing import TypeVar

import torch

from torch_mlir.ir import Module

# A type shared between the result of `OnnxBackend.compile` and the
# input to `OnnxBackend.load`. Each backend will likely have a
# different definition of this type.
CompiledArtifact = TypeVar('CompiledArtifact')

# A wrapper around a backend-specific loaded program representation
# that uniformly translates the `x.method(...)` interface expected of
# Torch modules into appropriate lower-level operations.
Invoker = TypeVar('Invoker')


class OnnxBackend(abc.ABC):
    """The interface to an ONNX backend.

    Backends are recommended to raise meaningful exceptions in case of error,
    ideally with easy reproduction instructions.
    """
    @abc.abstractmethod
    def compile(self, module: Module) -> CompiledArtifact:
        """Compile the provided MLIR module into a compiled artifact.

        The module adheres to the ONNX backend contract
        (see the VerifyOnnxBackendContract pass).

        The compiled artifact can be any type, but must be correctly
        interpreted by the `load` method.
        """

    @abc.abstractmethod
    def load(self, artifact: CompiledArtifact) -> Invoker:
        """Load the compiled artifact into a uniformly invokable form.

        The compiled artifact is the result of a previous call to `compile`.

        See the description of `Invoker` for the requirements on the returned
        type.
        """
