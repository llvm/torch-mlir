# -*- Python -*-
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# pylint: disable=no-member, no-name-in-module, invalid-name, missing-function-docstring, fixme

from typing import Iterable, Union
from torch.fx import GraphModule
from torch_mlir import ir
from torch_mlir.dialects import builtin
from .torch_mlir_types import TorchTensorType, PythonType

class Annotation:
    def __init__(self, types: Iterable[Union[TorchTensorType, type]]):
        self.types = list(map(lambda t:
                              PythonType(t) if isinstance(t, type) else t,
                              types))

    def __str__(self):
        result = f'Annotation instance with {len(self.types)} types\n'
        for e, type_ in enumerate(self.types):
            result += f'    Type of argument {e + 1}: {str(type_)}\n'
        return result

    def __iter__(self):
        return iter(self.types)


class AnnotationConverter:
    @staticmethod
    def to_mlir_array_attr(annotation: Annotation,
                           context: ir.Context) -> ir.ArrayAttr:
        dict_attrs = []
        for type_ in annotation:
            if not isinstance(type_, TorchTensorType):
                dict_attrs.append(ir.DictAttr.get({}, context=context))
                continue

            ir_type = type_.to_mlir(context)
            with context:
                type_attr = ir.TypeAttr.get(ir_type)
                dict_attr = ir.DictAttr.get({'torch.type_bound': type_attr})
                dict_attrs.append(dict_attr)

        return ir.ArrayAttr.get(dict_attrs, context=context)


def annotate_forward_args(module: GraphModule,
             types: Iterable[Union[TorchTensorType, type]]
             ) -> GraphModule:
    operands = filter(lambda node: node.op == 'placeholder', module.graph.nodes)
    for operand, type_ in zip(operands, types):
        if isinstance(type_, type):
            type_ = PythonType(type_)
        operand.update_kwarg('torch_mlir_type', type_)

    return module
