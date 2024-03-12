# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from torch_mlir.torchscript import TensorPlaceholder
from torch_mlir_e2e_test.annotations import TORCH_MLIR_ARG_ANNOTATIONS_ATTR_NAME

def convert_annotations_to_placeholders(forward_method):
    """Converts the annotations on a forward method into tensor placeholders.

    These placeholders are suitable for being passed to `torchscript.compile`.
    """
    annotations = getattr(forward_method, TORCH_MLIR_ARG_ANNOTATIONS_ATTR_NAME)
    placeholders = []
    # Skip the "self" annotation.
    for annotation in annotations[1:]:
        if not annotation[2]:
            raise ValueError(
                "Can only compile inputs annotated as having value semantics.")
        placeholders.append(TensorPlaceholder(annotation[0], annotation[1]))
    return placeholders
