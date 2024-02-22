# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

from typing import Any, Callable, Optional

import torch
import torch.export
import torch.nn as nn

from torch_mlir.extras.fx_importer import FxImporter
from torch_mlir.extras.fx_importer import SparsityMeta
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch_mlir.compiler_utils import run_pipeline_with_repro_report
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import (
    RefBackendLinalgOnTensorsBackend,
)


# All sparse layouts currently supported in torch.sparse.
SPARSE_LAYOUTS = [
    torch.sparse_coo,
    torch.sparse_csr,
    torch.sparse_csc,
    torch.sparse_bsr,
    torch.sparse_bsc,
]


def sparse_metadata(a: torch.Tensor) -> SparsityMeta:
    """
    Returns a meta data tuple for the given sparse tensor.

    NOTE: this will be fully replaced by fx graph SparseTensorMetadata
    """
    sparse_dim = a.sparse_dim()
    dense_dim = a.dense_dim()
    batch_dim = a.ndim - dense_dim - sparse_dim
    blocksize = None
    if a.layout is torch.sparse_coo:
        return SparsityMeta(
            a.layout,
            batch_dim,
            sparse_dim,
            dense_dim,
            blocksize,
            a.indices().dtype,
            a.indices().dtype,
        )
    elif a.layout is torch.sparse_csr or a.layout is torch.sparse_bsr:
        if a.layout is torch.sparse_bsr:
            blocksize = a.values().shape[batch_dim + 1 : batch_dim + 3]
        return SparsityMeta(
            a.layout,
            batch_dim,
            sparse_dim,
            dense_dim,
            blocksize,
            a.crow_indices().dtype,
            a.col_indices().dtype,
        )
    elif a.layout is torch.sparse_csc or a.layout is torch.sparse_bsc:
        if a.layout is torch.sparse_bsc:
            blocksize = a.values().shape[batch_dim + 1 : batch_dim + 3]
        return SparsityMeta(
            a.layout,
            batch_dim,
            sparse_dim,
            dense_dim,
            blocksize,
            a.ccol_indices().dtype,
            a.row_indices().dtype,
        )
    else:
        raise RuntimeError(f"Unsupported sparse layout for {a}")


def sparse_export(
    f: Callable, args: tuple[Any, ...], kwargs: Optional[dict[str, Any]] = None
) -> torch.export.ExportedProgram:
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
    resolved.
    """
    # Convert all arguments to dense.
    dargs = tuple(a.to_dense() if a.layout in SPARSE_LAYOUTS else a for a in args)
    mask = [a.layout in SPARSE_LAYOUTS for a in args]
    # Build the regular FX traced graph with only dense arguments
    # (the current version would crash otherwise, see issue above).
    prog = torch.export.export(f, dargs, kwargs, constraints=None)
    # Annotate sparse arguments in the graph. Note that we currently
    # only account for sparsity defined by the user inputs to the model.
    # TODO: support sparsity in model parameters (weights, biases)
    # TODO: propagate sparsity into the layers
    specs = prog.graph_signature.input_specs
    alen = len(specs)
    k = 0
    for i, node in enumerate(prog.graph.nodes):
        if i >= alen:
            break
        spec = specs[i]
        if spec.kind is torch.export.graph_signature.InputKind.USER_INPUT:
            if mask[k]:
                node.meta["sparsity"] = sparse_metadata(args[k])
            k = k + 1
    return prog


def export_and_import(f, *args, **kwargs):
    """This method implements Stella's importer, stripped down to essentials."""
    context = ir.Context()
    torch_d.register_dialect(context)
    fx_importer = FxImporter(context=context)
    prog = sparse_export(f, args, kwargs)
    fx_importer.import_frozen_program(prog)
    return fx_importer.module


def sparse_jit(f, *args, **kwargs):
    """This method compiles and runs the given callable using linalg backend."""
    # Import module and lower into Linalg IR.
    module = export_and_import(f, *args, *kwargs)
    run_pipeline_with_repro_report(
        module,
        "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
        "Lowering TorchFX IR -> Linalg IR",
        enable_ir_printing=False,
    )
    # Compile with reference Linalg backend.
    backend = RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(module)
    invoker = backend.load(compiled)
    # Prepare input parameters. Sparse input tensors are split into
    # their composite tensors. All PyTorch tensors are converted
    # to their backing numpy arrays.
    #
    # TODO: sparse output tensors
    #
    xargs = []
    for a in args:
        if a.layout is torch.sparse_coo:
            xargs.append(a.values().numpy())
            xargs.append(a.indices().numpy())
        elif a.layout is torch.sparse_csr or a.layout is torch.sparse_bsr:
            xargs.append(a.values().numpy())
            xargs.append(a.crow_indices().numpy())
            xargs.append(a.col_indices().numpy())
        elif a.layout is torch.sparse_csc or a.layout is torch.sparse_bsc:
            xargs.append(a.values().numpy())
            xargs.append(a.ccol_indices().numpy())
            xargs.append(a.row_indices().numpy())
        else:
            xargs.append(a.numpy())
    # Invoke.
    return invoker.main(*xargs)


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# CHECK-LABEL: test_sparse_sum
# CHECK:       #[[$CSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed), posWidth = 64, crdWidth = 64 }>
# CHECK:       func.func @main(
# CHECK-SAME:    %[[A:.*]]: !torch.vtensor<[64,64],f32,#[[$CSR]]>) -> !torch.vtensor<[],f32> {
# CHECK:         %[[N:.*]] = torch.constant.none
# CHECK:         %[[R:.*]] = torch.aten.sum %[[A]], %[[N]] : !torch.vtensor<[64,64],f32,#[[$CSR]]>, !torch.none -> !torch.vtensor<[],f32>
# CHECK:         return %[[R]] : !torch.vtensor<[],f32>
# CHECK:       }
#
# CHECK: torch.sparse = tensor(4096.)
# CHECK: torch.mlir   = 4096.0
#
def test_sparse_sum():
    class SumNet(torch.nn.Module):
        def __init__(self):
            super(SumNet, self).__init__()

        def forward(self, x):
            return x.sum()

    dense_input = torch.ones(64, 64)
    sparse_input = dense_input.to_sparse_csr()
    m = export_and_import(SumNet(), sparse_input)
    print(m)

    # Run it with PyTorch torch.sparse and with TORCH-MLIR sparse_jit.
    net = SumNet()
    res1 = net(sparse_input)
    res2 = sparse_jit(net, sparse_input)
    print("torch.sparse =", res1)
    print("torch.mlir   =", res2)


@run
# CHECK-LABEL: test_sparse_SpMV
# CHECK:       #[[$BSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 floordiv 2 : dense, d1 floordiv 2 : compressed, d0 mod 2 : dense, d1 mod 2 : dense), posWidth = 64, crdWidth = 64 }>
# CHECK:       func.func @main(
# CHECK-SAME:    %[[A:.*0]]: !torch.vtensor<[10,10],f32,#[[$BSR]]>,
# CHECK-SAME:    %[[B:.*1]]: !torch.vtensor<[10],f32>) -> !torch.vtensor<[10],f32> {
# CHECK:         %[[R:.*]] = torch.aten.mv %[[A]], %[[B]] : !torch.vtensor<[10,10],f32,#[[$BSR]]>, !torch.vtensor<[10],f32> -> !torch.vtensor<[10],f32>
# CHECK:         return %[[R]] : !torch.vtensor<[10],f32>
# CHECK:       }
def test_sparse_SpMV():
    class SpMVNet(torch.nn.Module):
        def __init__(self):
            super(SpMVNet, self).__init__()

        def forward(self, x, v):
            return torch.mv(x, v)

    dense_vector = torch.ones(10)
    dense_input = torch.ones(10, 10)
    sparse_input = dense_input.to_sparse_bsr(blocksize=(2, 2))
    m = export_and_import(SpMVNet(), sparse_input, dense_vector)
    print(m)


@run
# CHECK-LABEL: test_sparse_SpMM
# CHECK:       #[[$COO:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton), posWidth = 64, crdWidth = 64 }>
# CHECK:       func.func @main(
# CHECK-SAME:    %[[A:.*0]]: !torch.vtensor<[8,8],f32,#[[$COO]]>,
# CHECK-SAME:    %[[B:.*1]]: !torch.vtensor<[8,8],f32>) -> !torch.vtensor<[8,8],f32> {
# CHECK:         %[[R:.*]] = torch.aten.mm %[[A]], %[[B]] : !torch.vtensor<[8,8],f32,#[[$COO]]>, !torch.vtensor<[8,8],f32> -> !torch.vtensor<[8,8],f32>
# CHECK:         return %[[R]] : !torch.vtensor<[8,8],f32>
# CHECK:       }
#
# CHECK:        torch.sparse
# CHECK:        tensor({{\[}}[8., 8., 8., 8., 8., 8., 8., 8.],
# CHECK-COUNT-6:             [8., 8., 8., 8., 8., 8., 8., 8.],
# CHECK:                     [8., 8., 8., 8., 8., 8., 8., 8.]{{\]}})
# CHECK:        torch.mlir
# CHECK:        {{\[}}[8. 8. 8. 8. 8. 8. 8. 8.]
# CHECK-COUNT-6:      [8. 8. 8. 8. 8. 8. 8. 8.]
# CHECK:              [8. 8. 8. 8. 8. 8. 8. 8.]{{\]}}
#
def test_sparse_SpMM():
    class MatMulNet(torch.nn.Module):
        def __init__(self):
            super(MatMulNet, self).__init__()

        def forward(self, x, y):
            return torch.matmul(x, y)

    dense_input = torch.ones(8, 8)
    sparse_input = dense_input.to_sparse_coo()
    m = export_and_import(MatMulNet(), sparse_input, dense_input)
    print(m)

    # Run it with PyTorch torch.sparse and with TORCH-MLIR sparse_jit.
    # TODO: run with COO, right now only CSR works
    sparse_input = dense_input.to_sparse_csr()
    net = MatMulNet()
    res1 = net(sparse_input, dense_input)
    res2 = sparse_jit(net, sparse_input, dense_input)
    print("torch.sparse")
    print(res1)
    print("torch.mlir")
    print(res2)


@run
# CHECK-LABEL: test_sparse_eltwise
# CHECK:       #[[$BCSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : dense, d1 : dense, d2 : compressed), posWidth = 64, crdWidth = 64 }>
# CHECK:       func.func @main(
# CHECK-SAME:    %[[A:.*]]: !torch.vtensor<[8,4,2],f32,#[[$BCSR]]>) -> !torch.vtensor<[8,4,2],f32> {
# CHECK:         %[[R:.*]] = torch.aten.neg %arg0 : !torch.vtensor<[8,4,2],f32,#[[$BCSR]]> -> !torch.vtensor<[8,4,2],f32>
# CHECK:         return %[[R]] : !torch.vtensor<[8,4,2],f32>
# CHECK:       }
# CHECK:       #[[$CSRD:.*]] = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : dense, d1 : compressed, d2 : dense), posWidth = 64, crdWidth = 64 }>
# CHECK:       func.func @main(
# CHECK-SAME:    %[[A:.*]]: !torch.vtensor<[8,4,2],f32,#[[$CSRD]]>) -> !torch.vtensor<[8,4,2],f32> {
# CHECK:         %[[R:.*]] = torch.aten.neg %arg0 : !torch.vtensor<[8,4,2],f32,#[[$CSRD]]> -> !torch.vtensor<[8,4,2],f32>
# CHECK:         return %[[R]] : !torch.vtensor<[8,4,2],f32>
# CHECK:       }
#
# CHECK:        torch.sparse
# CHECK:        tensor(crow_indices=tensor([ 0,  4,  8, 12, 16, 20, 24, 28, 32]),
# CHECK:        col_indices=tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
# CHECK:                            2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]),
# CHECK:        values=tensor({{\[}}[ -1.,  -2.],
# CHECK:                            [ -3.,  -4.],
#                                   ...
# CHECK:                            [-63., -64.]{{\]}}), size=(8, 4, 2), nnz=32,
# CHECK:                       layout=torch.sparse_csr)
# CHECK:        torch.mlir
# CHECK:        {{\[\[}}[ -1.  -2.]
# CHECK:                [ -3.  -4.]
#                       ...
# CHECK:                [-61. -62.]
# CHECK:                [-63. -64.]{{\]\]}}
def test_sparse_eltwise():
    class EltNet(torch.nn.Module):
        def __init__(self):
            super(EltNet, self).__init__()

        def forward(self, x):
            return -x

    dense_input = torch.reshape(
        torch.arange(1, 65, dtype=torch.float32), shape=(8, 4, 2)
    )

    # This yields a **batched** CSR.
    sparse_input = dense_input.to_sparse_csr(dense_dim=0)
    m = export_and_import(EltNet(), sparse_input)
    print(m)

    # This yields a plain CSR with dense **sub**tensor
    sparse_input = dense_input.to_sparse_csr(dense_dim=1)
    m = export_and_import(EltNet(), sparse_input)
    print(m)

    # Run it with PyTorch torch.sparse and with TORCH-MLIR sparse_jit.
    #
    # TODO: note several issues that need to be fixed
    #  (1) since we do not propagate sparsity into elt-wise, MLIR returns dense result
    #  (2) for dense_dim=0, this will need a dense(batched) property
    sparse_input = dense_input.to_sparse_csr(dense_dim=1)
    net = EltNet()
    res1 = net(sparse_input)
    res2 = sparse_jit(net, sparse_input)
    print("torch.sparse")
    print(res1)
    print("torch.mlir")
    print(res2)
