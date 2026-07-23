# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

"""Helpers for annotating a Torch FX graph with MLIR attributes."""

from typing import Any, Dict, Optional

import torch
import torch.library


__all__ = [
    "annotate_and_pass_through",
    "AnnotateAndPassThrough",
]


_LIB: Optional[torch.library.Library] = None
if not hasattr(torch.ops.torch_mlir, "annotate_and_pass_through"):
    _LIB = torch.library.Library("torch_mlir", "DEF")
    _LIB.define(
        "annotate_and_pass_through(Tensor input, Dict(str, Any) annotations) -> Tensor"
    )
    _LIB.impl(
        "annotate_and_pass_through",
        lambda input, annotations: input.clone(),
        "CompositeExplicitAutograd",
    )
    _LIB.impl(
        "annotate_and_pass_through",
        lambda input, annotations: torch.empty_like(input),
        "Meta",
    )


def annotate_and_pass_through(
    input: torch.Tensor, annotations: Dict[str, Any]
) -> torch.Tensor:
    """A marker op that lowers `annotations` to MLIR attributes.

    The `annotations` dictionary for this op lowers to MLIR NamedAttributes in
    two ways:

    1. When `input` is an input node, the annotations lower to func.func
       arg_attrs for the corresponding argument.
    2. Otherwise, the annotations lower to discardable attributes on the
       op that defines the SSA value that `input` lowers to.
    """
    return torch.ops.torch_mlir.annotate_and_pass_through(input, annotations)


AnnotateAndPassThrough = annotate_and_pass_through
