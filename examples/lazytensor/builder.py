# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""
Translator from torch.jit.ScriptFunction to MLIR.

The following defines a function that take a torch.jit.ScriptFunction
and converts it into an MLIR module.

The expected use for this module is to use the function
`build_module(jit_function: torch.jit.ScriptFunction
              annotation: Annotation) -> ir.Module`
to convert the TorchScript function into MLIR using the `torch`
dialect.
"""

from typing import Optional

from torch.jit import ScriptFunction

from torch_mlir.dialects.torch.importer.jit_ir import ModuleBuilder
from torch_mlir.dialects.builtin import FuncOp
from torch_mlir import ir

from utils.annotator import AnnotationConverter as ac
from utils.annotator import Annotation

def _get_func_op_with_name(module: ir.Module, name: str) -> Optional[FuncOp]:
    with module.context:
        name_attr = ir.StringAttr.get(name)
    for op in module.body.operations:
        if isinstance(op, FuncOp) and op.name == name_attr:
            return op

    return None

def build_module(jit_function: ScriptFunction,
                 annotation: Annotation) -> ir.Module:
    """
    Translate input function into an MLIR module in the `torch` dialect.

    Parameters
    ----------
    jit_function: ScriptFunction
        Function in TorchScript IR to turn into MLIR.
    annotation: Annotation
        Annotation object representing the types of
        the operands of `jit_function`.

    Returns
    -------
    ir.Module
        Translation of the input module into an MLIR module
    """
    mb = ModuleBuilder()
    mb.import_function(jit_function)

    func_op = _get_func_op_with_name(mb.module, jit_function.name)
    assert func_op is not None, 'Unable to find FuncOp in new module. Make sure function was imported correctly into ModuleBuilder'

    arg_attrs = ac.to_mlir_array_attr(annotation, mb.context)
    func_op.attributes['arg_attrs'] = arg_attrs

    return mb.module
