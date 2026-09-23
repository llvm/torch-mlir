# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s

import unittest

import torch
import torch.nn.functional as F

from torch_mlir import ir
from torch_mlir.dialects import torch as torch_dialect
from torch_mlir.extras.fx_importer import FxImporter


class LinearWeight(torch.nn.Module):
    def __init__(self, nested, shared):
        super().__init__()
        self.nested = nested
        self.shared = shared
        owner = self
        if nested:
            self.layer = torch.nn.Module()
            owner = self.layer
        owner.register_parameter("weight", torch.nn.Parameter(torch.ones(3, 4)))

    def forward(self, x):
        weight = self.layer.weight if self.nested else self.weight
        result = F.linear(x, weight)
        if self.shared:
            result = result + F.linear(x + 1, weight)
        return result


class NestedBufferList(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Module()
        self.layer.register_buffer("value", torch.ones(2, 4))

    def forward(self, x):
        return torch.cat([x, self.layer.value, self.layer.value])


def export_graph(model):
    x = torch.randn(2, 4)
    graph = torch.export.export(model, (x,)).module()
    torch.testing.assert_close(graph(x), model(x))
    return graph


def import_graph(graph):
    with ir.Context() as context:
        torch_dialect.register_dialect(context)
        importer = FxImporter(context=context)
        importer.import_graph_module(graph)
        assert importer.module.operation.verify()
        return importer.module


class GetAttrTest(unittest.TestCase):
    def test_linear_weight(self):
        for nested in (False, True):
            for shared in (False, True):
                with self.subTest(nested=nested, shared=shared):
                    module = import_graph(export_graph(LinearWeight(nested, shared)))
                    ops = list(
                        module.body.operations[0].regions[0].blocks[0].operations
                    )
                    literals = [
                        op for op in ops if op.operation.name == "torch.vtensor.literal"
                    ]
                    linears = [
                        op for op in ops if op.operation.name == "torch.aten.linear"
                    ]
                    self.assertEqual(len(literals), 1)
                    self.assertEqual(len(linears), 2 if shared else 1)
                    for linear in linears:
                        self.assertEqual(linear.operands[1], literals[0].results[0])

    def test_nested_shared_list_operand(self):
        module = import_graph(export_graph(NestedBufferList()))
        ops = list(module.body.operations[0].regions[0].blocks[0].operations)
        literals = [op for op in ops if op.operation.name == "torch.vtensor.literal"]
        operands = next(
            op for op in ops if op.operation.name == "torch.prim.ListConstruct"
        )
        self.assertEqual(len(literals), 1)
        self.assertEqual(operands.operands[1], literals[0].results[0])
        self.assertEqual(operands.operands[2], literals[0].results[0])

    def test_missing_nested_attribute(self):
        graph = export_graph(LinearWeight(nested=True, shared=False))
        weight = next(node for node in graph.graph.nodes if node.op == "get_attr")
        weight.target = "layer.missing"
        with self.assertRaisesRegex(AssertionError, "layer.missing.*no such attribute"):
            import_graph(graph)


if __name__ == "__main__":
    unittest.main()
