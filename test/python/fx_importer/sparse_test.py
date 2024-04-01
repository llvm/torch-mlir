# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

from typing import Any, Callable, Optional, Tuple, Dict

import torch
import torch.nn as nn

from torch_mlir.extras.fx_importer import FxImporter
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch_mlir.compiler_utils import run_pipeline_with_repro_report
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import (
    RefBackendLinalgOnTensorsBackend,
)


def export_and_import(f, *args, **kwargs):
    """This method implements Stella's importer, stripped down to essentials."""
    context = ir.Context()
    torch_d.register_dialect(context)
    fx_importer = FxImporter(context=context)
    prog = torch.export.export(f, args, kwargs)
    fx_importer.import_frozen_program(prog)
    return fx_importer.module


def sparse_jit(f, *args, **kwargs):
    """This method compiles and runs the given callable using linalg backend."""
    # Import module and lower into Linalg IR.
    module = export_and_import(f, *args, *kwargs)
    run_pipeline_with_repro_report(
        module,
        (
            "builtin.module("
            "func.func(torch-decompose-complex-ops),"
            "torch-backend-to-linalg-on-tensors-backend-pipeline)"
        ),
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
            # Construct the additional position array required by MLIR with data
            # array([0, nnz]).
            xargs.append(torch.tensor([0, a._nnz()], dtype=a.indices().dtype).numpy())
            # Transform a tensor<ndim x nnz> into [tensor<nnz> x ndim] to conform
            # MLIR SoA COO representation.
            for idx in a.indices():
                xargs.append(idx.numpy())
            xargs.append(a.values().numpy())
        elif a.layout is torch.sparse_csr or a.layout is torch.sparse_bsr:
            xargs.append(a.crow_indices().numpy())
            xargs.append(a.col_indices().numpy())
            xargs.append(a.values().numpy())
        elif a.layout is torch.sparse_csc or a.layout is torch.sparse_bsc:
            xargs.append(a.ccol_indices().numpy())
            xargs.append(a.row_indices().numpy())
            xargs.append(a.values().numpy())
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

    net = SumNet()
    dense_input = torch.ones(64, 64)
    sparse_input = dense_input.to_sparse_csr()
    m = export_and_import(net, sparse_input)
    print(m)

    # Run it with PyTorch torch.sparse and with TORCH-MLIR sparse_jit.
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
#
# CHECK: torch.sparse = tensor([55., 55., 55., 55., 55., 55., 55., 55., 55., 55.])
# CHECK: torch.mlir   = [55. 55. 55. 55. 55. 55. 55. 55. 55. 55.]
#
def test_sparse_SpMV():
    class SpMVNet(torch.nn.Module):
        def __init__(self):
            super(SpMVNet, self).__init__()

        def forward(self, x, v):
            return torch.mv(x, v)

    net = SpMVNet()
    dense_vector = torch.arange(1, 11, dtype=torch.float32)
    dense_input = torch.ones(10, 10)
    sparse_input = dense_input.to_sparse_bsr(blocksize=(2, 2))
    m = export_and_import(net, sparse_input, dense_vector)
    print(m)

    # Run it with PyTorch torch.sparse and with TORCH-MLIR sparse_jit.
    res1 = net(sparse_input, dense_vector)
    res2 = sparse_jit(net, sparse_input, dense_vector)
    print("torch.sparse =", res1)
    print("torch.mlir   =", res2)


@run
# CHECK-LABEL: test_sparse_SpMM
# CHECK:       #[[$COO:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton(soa)), posWidth = 64, crdWidth = 64 }>
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

    net = MatMulNet()
    dense_input = torch.ones(8, 8)
    sparse_input = dense_input.to_sparse_coo()
    m = export_and_import(net, sparse_input, dense_input)
    print(m)

    # Run it with PyTorch torch.sparse and with TORCH-MLIR sparse_jit.
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
# CHECK: torch.sparse
# CHECK: tensor(crow_indices=tensor([ 0,  4,  8, 12, 16, 20, 24, 28, 32]),
# CHECK: col_indices=tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
# CHECK:                     2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]),
# CHECK: values=tensor({{\[}}[ -1.,  -2.],
#                            ...
# CHECK:                     [-63., -64.]{{\]}}), size=(8, 4, 2), nnz=32,
# CHECK: layout=torch.sparse_csr)
# CHECK: torch.mlir
# CHECK: {{\[\[}}[ -1.  -2.]
# CHECK:         [ -3.  -4.]
#                ...
# CHECK:         [-61. -62.]
# CHECK:         [-63. -64.]{{\]\]}}
#
def test_sparse_eltwise():
    class EltNet(torch.nn.Module):
        def __init__(self):
            super(EltNet, self).__init__()

        def forward(self, x):
            return -x

    net = EltNet()
    dense_input = torch.reshape(
        torch.arange(1, 65, dtype=torch.float32), shape=(8, 4, 2)
    )

    # This yields a **batched** CSR.
    sparse_input = dense_input.to_sparse_csr(dense_dim=0)
    m = export_and_import(net, sparse_input)
    print(m)

    # This yields a plain CSR with dense **sub**tensor
    sparse_input = dense_input.to_sparse_csr(dense_dim=1)
    m = export_and_import(net, sparse_input)
    print(m)

    # Run it with PyTorch torch.sparse and with TORCH-MLIR sparse_jit.
    #
    # TODO: note several issues that need to be fixed
    #  (1) since we do not propagate sparsity into elt-wise, MLIR returns dense result
    #  (2) for dense_dim=0, this will need a dense(batched) property
    sparse_input = dense_input.to_sparse_csr(dense_dim=1)
    res1 = net(sparse_input)
    res2 = sparse_jit(net, sparse_input)
    print("torch.sparse")
    print(res1)
    print("torch.mlir")
    print(res2)


@run
# CHECK-LABEL: test_sparse_coo3
# CHECK:       #[[$COO3:.*]] = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : compressed(nonunique), d1 : singleton(nonunique, soa), d2 : singleton(soa)), posWidth = 64, crdWidth = 64 }>
# CHECK:       func.func @main(
# CHECK-SAME:    %[[A:.*]]: !torch.vtensor<[10,20,30],f64,#sparse>) -> !torch.vtensor<[10,20,30],f64,#sparse> {
# CHECK:         %[[R:.*]] = torch.aten.relu %[[A]] : !torch.vtensor<[10,20,30],f64,#sparse> -> !torch.vtensor<[10,20,30],f64,#sparse>
# CHECK:         return %[[R]] : !torch.vtensor<[10,20,30],f64,#sparse>
# CHECK:       }
#
# CHECK: torch.sparse
# CHECK: tensor(indices=tensor([[ 0,  1,  1,  4,  9,  9],
# CHECK:                        [ 0,  1,  1,  5, 19, 19],
# CHECK:                        [ 0,  1,  3,  6, 28, 29]]),
# CHECK: values=tensor([   0.,    0.,    1.,    2.,    3., 1000.]),
# CHECK: size=(10, 20, 30), nnz=6, dtype=torch.float64, layout=torch.sparse_coo)
# CHECK: torch.mlir
# CHECK: tensor(indices=tensor([[ 0,  1,  1,  4,  9,  9],
# CHECK:                        [ 0,  1,  1,  5, 19, 19],
# CHECK:                        [ 0,  1,  3,  6, 28, 29]]),
# CHECK: values=tensor([   0.,    0.,    1.,    2.,    3., 1000.]),
#
def test_sparse_coo3():
    class COO3Net(torch.nn.Module):
        def __init__(self):
            super(COO3Net, self).__init__()
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(x)

    net = COO3Net()

    # Direct 3-dim COO construction.
    idx = torch.tensor([[0, 1, 1, 4, 9, 9], [0, 1, 1, 5, 19, 19], [0, 1, 3, 6, 28, 29]])
    val = torch.tensor([-1000.0, -1.0, 1.0, 2.0, 3.0, 1000.0], dtype=torch.float64)
    sparse_input = torch.sparse_coo_tensor(idx, val, size=[10, 20, 30])

    m = export_and_import(net, sparse_input)
    print(m)

    # Run it with PyTorch torch.sparse and with TORCH-MLIR sparse_jit.
    res1 = net(sparse_input)
    res2 = sparse_jit(net, sparse_input)
    print("torch.sparse")
    print(res1)
    print("torch.mlir")
    print(res2)
