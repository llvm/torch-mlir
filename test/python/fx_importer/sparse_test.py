# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.export
import torch.nn as nn

from torch_mlir.extras.fx_importer import FxImporter
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d


# All sparse layouts currently supported in torch.sparse.
SPARSE_LAYOUTS = [
    torch.sparse_coo,
    torch.sparse_csr,
    torch.sparse_csc,
    torch.sparse_bsr,
    torch.sparse_bsc
]


def sparse_export(f: Callable,
                  args: Tuple[Any, ...],
                  kwargs: Optional[Dict[str, Any]] = None) -> torch.export.ExportedProgram:
    """
    This is a ***temporary*** wrapper around `torch.export.export`
    that eventually should be removed and simply replaced by the
    standard API for exporting traced graphs.

    But until issue

      https://github.com/pytorch/pytorch/pull/117907

    is addressed, this wrapper provides support for the sparse
    tensor types by first converting all operands to dense tensors,
    building the traced graph as for the dense case, and then
    annotation sparse parameters with their actual sparse layout
    attributes. This temporary solution accelerates testing
    torch-mlir with PyTorch sparse tensors until the issue is
    resovled.
    """
    # Convert all arguments to dense.
    dargs = tuple( a.to_dense() if a.layout in SPARSE_LAYOUTS else a for a in args )
    mask = [ a.layout in SPARSE_LAYOUTS for a in args ]
    # Build the regular FX traced graph with only dense arguments
    # (the current version would crash otherwise, see issue above).
    prog = torch.export.export(f, dargs, kwargs, constraints=None)
    # Annotate sparse arguments in the graph.
    alen = len(args)
    for i, node in enumerate(prog.graph.nodes):
      if node.op == "placeholder" and i < alen and mask[i]:
        node.meta['sparsity'] = args[i].layout
    # TODO: annotate inputs to change calling conventions!
    return prog


def export_and_import(f, *args, **kwargs):
    """This method implements Stella's importer, stripped down to essentials."""
    context = ir.Context()
    torch_d.register_dialect(context)
    fx_importer = FxImporter(context=context)
    prog = sparse_export(f, args, kwargs)
    fx_importer.import_frozen_exported_program(prog)
    return fx_importer.module_op


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# CHECK-LABEL: test_sparse_sum
# CHECK:       #[[$CSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
# CHECK:       func.func @main(
# CHECK-SAME:    %[[A:.*]]: !torch.vtensor<[64,64],f32,#[[$CSR]]>) -> !torch.vtensor<[],f32> {
# CHECK:         %[[N:.*]] = torch.constant.none
# CHECK:         %[[R:.*]] = torch.aten.sum %[[A]], %[[N]] : !torch.vtensor<[64,64],f32,#[[$CSR]]>, !torch.none -> !torch.vtensor<[],f32>
# CHECK:         return %[[R]] : !torch.vtensor<[],f32>
# CHECK:       }
def test_sparse_sum():

    class SumNet(torch.nn.Module):

        def __init__(self):
            super(SumNet, self).__init__()

        def forward(self, x):
            return x.sum()


    dense_input  = torch.ones(64, 64)
    sparse_input = dense_input.to_sparse_csr()
    m = export_and_import(SumNet(), sparse_input)
    print(m)


@run
# CHECK-LABEL: test_sparse_SpMM
# CHECK:       #[[$COO:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton) }>
# CHECK:       func.func @main(
# CHECK-SAME:    %[[A:.*0]]: !torch.vtensor<[64,64],f32,#[[$COO]]>,
# CHECK-SAME:    %[[B:.*1]]: !torch.vtensor<[64,64],f32>) -> !torch.vtensor<[64,64],f32> {
# CHECK:         %[[R:.*]] = torch.aten.mm %[[A]], %[[B]] : !torch.vtensor<[64,64],f32,#[[$COO]]>, !torch.vtensor<[64,64],f32> -> !torch.vtensor<[64,64],f32>
# CHECK:         return %[[R]] : !torch.vtensor<[64,64],f32>
# CHECK:       }
def test_sparse_SpMM():

    class MatMulNet(torch.nn.Module):

        def __init__(self):
            super(MatMulNet, self).__init__()

        def forward(self, x, y):
          return torch.matmul(x, y)


    dense_input  = torch.ones(64, 64)
    sparse_input = dense_input.to_sparse_coo()
    m = export_and_import(MatMulNet(), sparse_input, dense_input)
    print(m)
