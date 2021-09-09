# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.
#
# pylint: disable=no-member, no-name-in-module, invalid-name, missing-function-docstring, fixme

from typing import Iterable, Union
from torch.fx import GraphModule
from .torch_mlir_types import TorchTensorType, PythonType

def annotate_forward_args(module: GraphModule,
             types: Iterable[Union[TorchTensorType, type]]
             ) -> GraphModule:
    operands = filter(lambda node: node.op == 'placeholder', module.graph.nodes)
    for operand, type_ in zip(operands, types):
        if isinstance(type_, type):
            type_ = PythonType(type_)
        operand.update_kwarg('torch_mlir_type', type_)

    return module
