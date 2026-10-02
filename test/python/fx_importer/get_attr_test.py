# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch
import torch.nn.functional as F

from torch_mlir import ir
from torch_mlir.dialects import torch as torch_dialect
from torch_mlir.extras.fx_importer import FxImporter


def run(f):
    print(f.__name__)
    f()
    print()
    return f


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
    # The frozen-program path used by fx.export_and_import lifts parameters and
    # buffers to placeholders, so it does not exercise get_attr import.
    with ir.Context() as context:
        torch_dialect.register_dialect(context)
        importer = FxImporter(context=context)
        importer.import_graph_module(graph)
        assert importer.module.operation.verify()
        return importer.module


@run
# CHECK-LABEL: test_flat_weight
# CHECK: %[[WEIGHT:.+]] = torch.vtensor.literal
# CHECK-NOT: torch.vtensor.literal
# CHECK: torch.aten.linear %{{.*}}, %[[WEIGHT]],
# CHECK-NOT: torch.vtensor.literal
# CHECK-NOT: torch.aten.linear
# CHECK: return
def test_flat_weight():
    print(import_graph(export_graph(LinearWeight(nested=False, shared=False))))


@run
# CHECK-LABEL: test_nested_weight
# CHECK: %[[WEIGHT:.+]] = torch.vtensor.literal
# CHECK-NOT: torch.vtensor.literal
# CHECK: torch.aten.linear %{{.*}}, %[[WEIGHT]],
# CHECK-NOT: torch.vtensor.literal
# CHECK-NOT: torch.aten.linear
# CHECK: return
def test_nested_weight():
    print(import_graph(export_graph(LinearWeight(nested=True, shared=False))))


@run
# CHECK-LABEL: test_flat_shared_weight
# CHECK: %[[WEIGHT:.+]] = torch.vtensor.literal
# CHECK-NOT: torch.vtensor.literal
# CHECK: torch.aten.linear %{{.*}}, %[[WEIGHT]],
# CHECK-NOT: torch.vtensor.literal
# CHECK: torch.aten.linear %{{.*}}, %[[WEIGHT]],
# CHECK-NOT: torch.vtensor.literal
# CHECK-NOT: torch.aten.linear
# CHECK: return
def test_flat_shared_weight():
    print(import_graph(export_graph(LinearWeight(nested=False, shared=True))))


@run
# CHECK-LABEL: test_nested_shared_weight
# CHECK: %[[WEIGHT:.+]] = torch.vtensor.literal
# CHECK-NOT: torch.vtensor.literal
# CHECK: torch.aten.linear %{{.*}}, %[[WEIGHT]],
# CHECK-NOT: torch.vtensor.literal
# CHECK: torch.aten.linear %{{.*}}, %[[WEIGHT]],
# CHECK-NOT: torch.vtensor.literal
# CHECK-NOT: torch.aten.linear
# CHECK: return
def test_nested_shared_weight():
    print(import_graph(export_graph(LinearWeight(nested=True, shared=True))))


@run
# CHECK-LABEL: test_nested_shared_list_operand
# CHECK: %[[VALUE:.+]] = torch.vtensor.literal
# CHECK-NOT: torch.vtensor.literal
# CHECK: torch.prim.ListConstruct %{{.*}}, %[[VALUE]], %[[VALUE]]
# CHECK-NOT: torch.vtensor.literal
# CHECK: return
def test_nested_shared_list_operand():
    print(import_graph(export_graph(NestedBufferList())))


@run
# CHECK-LABEL: test_missing_nested_attribute
# CHECK: layer.missing rejected
def test_missing_nested_attribute():
    graph = export_graph(LinearWeight(nested=True, shared=False))
    weight = next(node for node in graph.graph.nodes if node.op == "get_attr")
    weight.target = "layer.missing"
    try:
        import_graph(graph)
    except AssertionError as exc:
        assert "layer.missing" in str(exc) and "no such attribute" in str(exc)
        print("layer.missing rejected")
    else:
        raise AssertionError("Missing nested attribute was imported")
