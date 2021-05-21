#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List, Optional, Tuple

import torch

import torch_mlir

# Decorators

# Currently, these decorators are very low-level and map 1:1 with
# methods on `torch_mlir.ClassAnnotator`. Eventually, we expect there to
# be a more elaborate Python layer which allows all the different annotations
# to be expressed conveniently and gives clearer error reports when
# the annotations aren't acceptable.


def export(fn):
    """Decorator that tells the npcomp compiler that a method is exported.

    By default, no methods are exported, which is very important for
    the compiler, because otherwise most Torch programs consist of a sea
    of tiny exported functions with no rank or dtype information
    (see `annotate_args`), which the compiler cannot do much with.

    Note that this is different from `torch.jit.export`, which controls
    which methods are scripted in the first place. For non-`forward` methods,
    using this decorator usually means you also need `torch.jit.export`.
    Conceptually, this decorator is annotating the scripted module, but is
    applied to the original `torch.nn.Module` for convenience.
    """
    fn._npcomp_export = True
    return fn


ArgAnnotation = Tuple[List[int], torch.dtype]


# TODO: Replace with py3 extended argument annotations when available.
# See https://www.python.org/dev/peps/pep-0593/
def annotate_args(annotations: List[Optional[ArgAnnotation]]):
    """Decorator that tells the npcomp compiler information about arguments.

    The `annotations` should be a list of the same length as the number of
    argument to the method (including `self`). Each list entry is either:
    - None, corresponding to providing the compiler with no information.
    - A 2-tuple consisting of a shape and a dtype, such as
      `([2, 3, 4], torch.float32)`. A dimension with an unknown size can be
      indicated by using `-1` as the size. This provides the compiler a
      guarantee that the argument will always dynamically have the described
      shape and dtype.
    """

    # TODO: Check the number of arguments matches the number of arg annotations.
    def decorator(fn):
        fn._npcomp_arg_annotations = annotations
        return fn

    return decorator


# Utilities for extracting decorated information into torch_mlir.ClassAnnotator.


def _recursively_extract_annotations(
        module: torch.nn.Module, scripted: torch.jit.ScriptModule,
        class_annotator: torch_mlir.ClassAnnotator):
    assert module.__class__.__name__ == scripted.original_name, "script module does not come from specified module"

    # Extract information on methods.
    for method_name, scripted_method in scripted.__dict__.items():
        if not isinstance(scripted_method, torch.ScriptMethod):
            continue
        method = getattr(module, method_name)
        if hasattr(method, '_npcomp_export'):
            class_annotator.exportPath(scripted._c._type(), [method_name])
        if hasattr(method, '_npcomp_arg_annotations'):
            class_annotator.annotateArgs(
                scripted._c._type(), [method_name],
                method._npcomp_arg_annotations)
    # Recurse.
    for name, child in module.named_children():
        scripted_child = getattr(scripted, name)
        _recursively_extract_annotations(child, scripted_child,
                                         class_annotator)


def extract_annotations(program: torch.nn.Module,
                        scripted: torch.jit.ScriptModule,
                        class_annotator: torch_mlir.ClassAnnotator):
    """Populate the ClassAnnotator with annotations extracted from `program`."""
    class_annotator.exportNone(scripted._c._type())
    _recursively_extract_annotations(program, scripted, class_annotator)
