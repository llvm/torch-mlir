# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

from typing import Any, Callable, Optional, Tuple, Dict

import torch
import torch.export
import torch.nn as nn
import numpy as np

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
            a._indices().dtype,
            a._indices().dtype,
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
    f: Callable, args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]] = None
) -> torch.export.ExportedProgram:
    """
    This is a ***temporary*** wrapper around `torch.export.export`
    that eventually should be removed and simply replaced by the
    standard API for exporting traced graphs.

    But until issue

      https://github.com/pytorch/pytorch/pull/117907

    is addressed, this wrapper provides support for the sparse
    tensor types by first converting all operands to dense tensors,
    building the traced graph as for the dense case, then annotating
    sparse parameters with their actual sparse layout attributes,
    followed by some simple propagation rules. This temporary solution
    accelerates testing torch-mlir with PyTorch sparse tensors until
    the issue is resolved upstream.
    """
    # Convert all arguments to dense.
    dargs = tuple(a.to_dense() if a.layout in SPARSE_LAYOUTS else a for a in args)
    mask = [a.layout in SPARSE_LAYOUTS for a in args]
    # Build the regular FX traced graph with only dense arguments
    # (the current version would crash otherwise, see issue above).
    prog = torch.export.export(f, dargs, kwargs)
    # Annotate sparse arguments in the graph and apply some very
    # basic propagation rules for sparsity.
    specs = prog.graph_signature.input_specs
    alen = len(specs)
    k = 0
    for i, node in enumerate(prog.graph.nodes):
        if node.op == "placeholder":
            # Argument.
            spec = specs[i]
            if spec.kind is torch.export.graph_signature.InputKind.USER_INPUT:
                if mask[k]:
                    node.meta["sparsity"] = sparse_metadata(args[k])
                k = k + 1
        elif node.op == "call_function":
            # TODO: use upstream _opname implementation when available
            opname = node.target._schema.name.split("::")[1]
            # Zero preserving elt-wise unary op.
            if opname in {"abs", "neg", "relu", "sin"}:
                node.meta["sparsity"] = node.args[0].meta.get("sparsity", None)
            elif opname == "_to_sparse":
                dim = len(node.meta.get("val").shape)
                node.meta["sparsity"] = SparsityMeta(
                    torch.sparse_coo, 0, dim, 0, None, torch.int64, torch.int64
                )
            # TODO: Uncomment this to hack sparsity into the network.
            # elif opname == "_to_dense":
            #     # hack (assumes we never really want the to_dense for now)
            #     node.meta["sparsity"] = node.args[0].meta.get("sparsity", None)
            elif opname == "select" and node.args[0].meta.get("sparsity", None):
                dim = len(node.meta.get("val").shape)
                node.meta["sparsity"] = SparsityMeta(
                    torch.sparse_coo, 0, dim, 0, None, torch.int64, torch.int64
                )
            elif opname == "stack" and node.args[0][0].meta.get("sparsity", None):
                dim = len(node.meta.get("val").shape)
                node.meta["sparsity"] = SparsityMeta(
                    torch.sparse_coo, 0, dim - 1, 1, None, torch.int64, torch.int64
                )
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
    module = export_and_import(f, *args, **kwargs)
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
    xargs = []
    # Prepare the buffer parameters (assume all dense).
    # TODO: filters out scalar arguments, anything else?
    params = dict(f.named_buffers(remove_duplicate=True))
    params_flat, params_spec = torch.utils._pytree.tree_flatten(params)
    for p in params_flat:
        if len(p.shape) > 0:
            xargs.append(p.numpy())
    # Prepare input parameters. Sparse input tensors are split into
    # their composite tensors. All PyTorch tensors are converted
    # to their backing numpy arrays. Note that the output consists
    # of numpy arrays as well, which can trivially be reconstructed
    # into PyTorch tensors (dense and sparse).
    for a in args:
        if a.layout is torch.sparse_coo:
            # Construct the additional position array required by MLIR with data
            # array([0, nnz]). The COO format always uses int64 indices.
            xargs.append(np.array([0, a._nnz()], dtype=np.int64))
            # Transform a tensor<ndim x nnz> into ndim x tensor<nnz> to conform
            # to the MLIR SoA COO representation.
            for idx in a._indices():
                xargs.append(idx.numpy())
            xargs.append(a._values().numpy())
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
#
# CHECK-LABEL: test_sparse_id
# CHECK:       #[[$COO:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton(soa)), posWidth = 64, crdWidth = 64 }>
# CHECK:       func.func @main(
# CHECK-SAME:    %[[A:.*]]: !torch.vtensor<[10,20],f64,#[[$COO]]>) -> !torch.vtensor<[10,20],f64,#[[$COO]]> {
# CHECK:         return %[[A]] : !torch.vtensor<[10,20],f64,#[[$COO]]>
# CHECK:       }
#
# CHECK: torch.sparse
# CHECK:   tensor(indices=tensor({{\[}}[ 0,  1,  2,  9],
# CHECK:                               [ 0,  1, 10, 19]{{\]}}),
# CHECK:          values=tensor([-1000.,    -1.,     1.,  1000.]),
# CHECK:          size=(10, 20), nnz=4, dtype=torch.float64, layout=torch.sparse_coo)
# CHECK: torch.mlir
# CHECK:   [0 4]
# CHECK:   [0 1 2 9]
# CHECK:   [ 0  1 10 19]
# CHECK:   [-1000.    -1.     1.  1000.]
#
def test_sparse_id():
    class IdNet(torch.nn.Module):
        def __init__(self):
            super(IdNet, self).__init__()

        def forward(self, x):
            return x

    net = IdNet()
    idx = torch.tensor([[0, 1, 2, 9], [0, 1, 10, 19]])
    val = torch.tensor([-1000.0, -1.0, 1.0, 1000.0], dtype=torch.float64)
    sparse_input = torch.sparse_coo_tensor(idx, val, size=[10, 20])
    m = export_and_import(net, sparse_input)
    print(m)

    # Run it with PyTorch torch.sparse and with TORCH-MLIR sparse_jit.
    res1 = net(sparse_input)
    res2 = sparse_jit(net, sparse_input)
    print("torch.sparse")
    print(res1)
    print("torch.mlir")
    print(res2[0])
    print(res2[1])
    print(res2[2])
    print(res2[3])


@run
#
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
#
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
#
# CHECK: torch.sparse
# CHECK:   tensor({{\[}}[8., 8., 8., 8., 8., 8., 8., 8.],
# CHECK-COUNT-6:        [8., 8., 8., 8., 8., 8., 8., 8.],
# CHECK:                [8., 8., 8., 8., 8., 8., 8., 8.]{{\]}})
# CHECK: torch.mlir
# CHECK:          {{\[}}[8. 8. 8. 8. 8. 8. 8. 8.]
# CHECK-COUNT-6:        [8. 8. 8. 8. 8. 8. 8. 8.]
# CHECK:                [8. 8. 8. 8. 8. 8. 8. 8.]{{\]}}
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
    # print(m)

    # Run it with PyTorch torch.sparse and with TORCH-MLIR sparse_jit.
    res1 = net(sparse_input, dense_input)
    res2 = sparse_jit(net, sparse_input, dense_input)
    print("torch.sparse")
    print(res1)
    print("torch.mlir")
    print(res2)


@run
#
# CHECK-LABEL: test_sparse_eltwise
# CHECK:       #[[$CSRD:.*]] = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : dense, d1 : compressed, d2 : dense), posWidth = 64, crdWidth = 64 }>
# CHECK:       func.func @main(
# CHECK-SAME:    %[[A:.*]]: !torch.vtensor<[4,2,2],f32,#[[$CSRD]]>) -> !torch.vtensor<[4,2,2],f32,#[[$CSRD]]> {
# CHECK:         %[[R:.*]] = torch.aten.neg %[[A]] : !torch.vtensor<[4,2,2],f32,#[[$CSRD]]> -> !torch.vtensor<[4,2,2],f32,#[[$CSRD]]>
# CHECK:         return %[[R]] : !torch.vtensor<[4,2,2],f32,#[[$CSRD]]>
# CHECK:       }
# CHECK:       #[[$BCSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : batch, d1 : dense, d2 : compressed), posWidth = 64, crdWidth = 64 }>
# CHECK:       func.func @main(
# CHECK-SAME:    %[[A:.*]]: !torch.vtensor<[4,2,2],f32,#[[$BCSR]]>) -> !torch.vtensor<[4,2,2],f32,#[[$BCSR]]> {
# CHECK:         %[[R:.*]] = torch.aten.neg %[[A]] : !torch.vtensor<[4,2,2],f32,#[[$BCSR]]> -> !torch.vtensor<[4,2,2],f32,#[[$BCSR]]>
# CHECK:         return %[[R]] : !torch.vtensor<[4,2,2],f32,#[[$BCSR]]>
# CHECK:       }
#
# CHECK: torch.sparse
# CHECK:   tensor(crow_indices=tensor([0, 2, 4, 6, 8]),
# CHECK:          col_indices=tensor([0, 1, 0, 1, 0, 1, 0, 1]),
# CHECK:          values=tensor({{\[}}[ -1.,  -2.],
# CHECK:                              [ -3.,  -4.],
# CHECK:                              [ -5.,  -6.],
# CHECK:                              [ -7.,  -8.],
# CHECK:                              [ -9., -10.],
# CHECK:                              [-11., -12.],
# CHECK:                              [-13., -14.],
# CHECK:                              [-15., -16.]{{\]}}), size=(4, 2, 2), nnz=8,
# CHECK:                              layout=torch.sparse_csr)
# CHECK: torch.mlir
# CHECK:   [0 2 4 6 8]
# CHECK:   [0 1 0 1 0 1 0 1]
# CHECK:   [ -1.  -2.  -3.  -4.  -5.  -6.  -7.  -8.  -9. -10. -11. -12. -13. -14.
# CHECK:    -15. -16.]
# CHECK: torch.mlir.batch
#
def test_sparse_eltwise():
    class EltNet(torch.nn.Module):
        def __init__(self):
            super(EltNet, self).__init__()

        def forward(self, x):
            return -x

    net = EltNet()
    dense_input = torch.reshape(
        torch.arange(1, 17, dtype=torch.float32), shape=(4, 2, 2)
    )

    # This yields a plain CSR with dense **sub**tensor
    sparse_input = dense_input.to_sparse_csr(dense_dim=1)
    m = export_and_import(net, sparse_input)
    print(m)

    # This yields a **batched** CSR.
    batch_input = dense_input.to_sparse_csr(dense_dim=0)
    m = export_and_import(net, batch_input)
    print(m)

    # Run it with PyTorch torch.sparse and with TORCH-MLIR sparse_jit.
    res1 = net(sparse_input)
    res2 = sparse_jit(net, sparse_input)
    # TODO: make this work
    # res3 = sparse_jit(net, batch_input)
    print("torch.sparse")
    print(res1)
    print("torch.mlir")
    print(res2[0])
    print(res2[1])
    print(res2[2])
    print("torch.mlir.batch")


@run
#
# CHECK-LABEL: test_sparse_coo3
# CHECK:       #[[$COO3:.*]] = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : compressed(nonunique), d1 : singleton(nonunique, soa), d2 : singleton(soa)), posWidth = 64, crdWidth = 64 }>
# CHECK:       func.func @main(
# CHECK-SAME:    %[[A:.*]]: !torch.vtensor<[10,20,30],f64,#[[$COO3]]>) -> !torch.vtensor<[10,20,30],f64,#[[$COO3]]> {
# CHECK:         %[[R:.*]] = torch.aten.relu %[[A]] : !torch.vtensor<[10,20,30],f64,#[[$COO3]]> -> !torch.vtensor<[10,20,30],f64,#[[$COO3]]>
# CHECK:         return %[[R]] : !torch.vtensor<[10,20,30],f64,#[[$COO3]]>
# CHECK:       }
#
# CHECK: torch.sparse
# CHECK:   tensor(indices=tensor({{\[}}[ 0,  1,  1,  4,  9,  9],
# CHECK:                               [ 0,  1,  1,  5, 19, 19],
# CHECK:                               [ 0,  1,  3,  6, 28, 29]{{\]}}),
# CHECK:          values=tensor([   0.,    0.,    1.,    2.,    3., 1000.]),
# CHECK:          size=(10, 20, 30), nnz=6, dtype=torch.float64, layout=torch.sparse_coo)
# CHECK: torch.mlir
# CHECK:  [0 6]
# CHECK:  [0 1 1 4 9 9]
# CHECK:  [ 0  1  1  5 19 19]
# CHECK:  [ 0  1  3  6 28 29]
# CHECK:  [   0.    0.    1.    2.    3. 1000.]
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
    print(res2[0])
    print(res2[1])
    print(res2[2])
    print(res2[3])
    print(res2[4])


@run
#
# CHECK: torch.sparse
# CHECK:   tensor(indices=tensor({{\[}}[0, 0, 0, 0, 1, 1, 1, 1],
# CHECK:                               [0, 0, 1, 1, 0, 0, 1, 1],
# CHECK:                               [0, 1, 0, 1, 0, 1, 0, 1]{{\]}}),
# CHECK:      values=tensor([1., 1., 1., 1., 1., 1., 1., 1.]),
# CHECK:      size=(2, 2, 2), nnz=8, layout=torch.sparse_coo)
#
def test_sparse_activation():
    class SparseActivationCOO(torch.nn.Module):
        def forward(self, x):
            return x.to_sparse()

    net = SparseActivationCOO()
    x = torch.ones(2, 2, 2)
    m = export_and_import(net, x)
    # print(m)

    # Run it with PyTorch torch.sparse and with TORCH-MLIR sparse_jit.
    res1 = net(x)
    # res2 = sparse_jit(net, x)
    print("torch.sparse")
    print(res1)
    # print("torch.mlir")
    # print(res2[0])
    # print(res2[1])
    # print(res2[2])
    # print(res2[3])
    # print(res2[4])


@run
#
# CHECK-LABEL: test_sparse_network
# CHECK:       func.func @main(
# CHECK-SAME:    %[[A:.*]]: !torch.vtensor<[2,3,8,8],f32>) -> !torch.vtensor<[8],f32> {
#                ... lots of IR ...
# CHECK-COUNT-15: torch.aten.mul.Tensor
#                ... lots of IR ...
# CHECK:        }
#
# CHECK: torch.sparse
# CHECK:   tensor([ 0., 11.,  9., 11., 13., 11., 10., 12.])
#
def test_sparse_network():
    def spike(input):
        return (input >= 0).float()

    def sqSum(input):
        return (input * input).sum()

    class LIF(nn.Module):
        def __init__(self):
            super(LIF, self).__init__()
            self.thresh = 1.0
            self.decay = 0.5
            self.act = spike

        def forward(self, X):
            """A filter that yields a binary-valued sparse tensor."""
            mem = 0
            spike_pot = []
            T = X.size(-1)
            for t in range(T):
                mem = mem * self.decay + X[..., t]
                spike = self.act(mem - self.thresh)
                spike = spike.to_sparse().to_dense()  # prop hack
                mem = mem * (1.0 - spike)
                spike_pot.append(spike)
            spike_pot = torch.stack(spike_pot, dim=-1)
            return spike_pot

    class tdLayer(nn.Module):
        def __init__(self, layer):
            super(tdLayer, self).__init__()
            self.layer = layer

        def forward(self, X):
            T = X.size(-1)
            out = []
            for t in range(T):
                m = self.layer(X[..., t])
                out.append(m)
            out = torch.stack(out, dim=-1)
            return out

    class Block(nn.Module):
        def __init__(self):
            super(Block, self).__init__()
            self.spike = LIF()
            self.layer = tdLayer(sqSum)

        def forward(self, X):
            out = self.spike(X)
            out = self.layer(out)
            return out

    net = Block()

    # Get a random (but reproducible) input, so that a
    # general sparse tensor appears after LIF.
    torch.manual_seed(0)
    x = torch.rand(2, 3, 8, 8)
    m = export_and_import(net, x)
    print(m)

    # Run it with PyTorch torch.sparse and with TORCH-MLIR sparse_jit.
    res1 = net(x)
    # res2 = sparse_jit(net, x)
    print("torch.sparse")
    print(res1)
    # print("torch.mlir")
    # print(res2)


@run
#
# CHECK: torch.sparse
# CHECK:   tensor({{\[}}[0.3342, 0.5173, 0.0596, 0.0889],
# CHECK:                [0.1321, 0.2724, 0.2105, 0.3851],
# CHECK:                [0.2478, 0.3439, 0.1898, 0.2185],
# CHECK:                [0.0222, 0.1683, 0.2928, 0.5167]{{\]}})
# CHECK: torch.mlir
#
def test_sparse_feature_scaling():
    class Scale(nn.Module):
        def forward(self, F):
            sum_vector = torch.sum(F, dim=1)
            reciprocal_vector = 1 / sum_vector
            reciprocal_vector[reciprocal_vector == float("inf")] = 0
            scaling_diagonal = torch.diag(reciprocal_vector).to_sparse()
            return scaling_diagonal @ F

    net = Scale()

    # Get a random (but reproducible) features input.
    torch.manual_seed(0)
    f = torch.rand(4, 4)
    m = export_and_import(net, f)
    # print(m)

    # Run it with PyTorch torch.sparse and with TORCH-MLIR sparse_jit.
    res1 = net(f)
    # TODO: make this work
    # res2 = sparse_jit(net, f)
    print("torch.sparse")
    print(res1)
    print("torch.mlir")


@run
#
# CHECK-LABEL: test_sparse_gcn
# CHECK:       #[[$COO:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton(soa)), posWidth = 64, crdWidth = 64 }>
# CHECK:       func.func @main(
# CHECK-SAME:    %[[A:.*]]: !torch.vtensor<[4,4],f32>,
# CHECK-SAME:    %[[B:.*]]: !torch.vtensor<[4,4],f32,#[[$COO]]>) -> !torch.vtensor<[4,4],f32> {
# CHECK:         %[[LIT:.*]] = torch.vtensor.literal(dense_resource<torch_tensor_4_4_torch.float32> : tensor<4x4xf32>) : !torch.vtensor<[4,4],f32>
# CHECK:         %[[MM:.*]] = torch.aten.mm %[[A]], %[[LIT]] : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f32> -> !torch.vtensor<[4,4],f32>
# CHECK:         %[[SMM:.*]] = torch.aten.mm %[[B]], %[[MM]] : !torch.vtensor<[4,4],f32,#sparse>, !torch.vtensor<[4,4],f32> -> !torch.vtensor<[4,4],f32>
# CHECK:         %[[BIAS:.*]] = torch.vtensor.literal(dense_resource<torch_tensor_4_torch.float32> : tensor<4xf32>) : !torch.vtensor<[4],f32>
# CHECK:         %[[ONE:.*]] = torch.constant.int 1
# CHECK:         %[[R:.*]] = torch.aten.add.Tensor %[[SMM]], %[[BIAS]], %[[ONE]] : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[4,4],f32>
# CHECK          return %[[R]] : !torch.vtensor<[4,4],f32>
# CHECK:        }
#
# CHECK: torch.sparse
# CHECK:   tensor({{\[}}[4.4778, 4.4778, 4.4778, 4.4778],
# CHECK:                [5.7502, 5.7502, 5.7502, 5.7502],
# CHECK:                [4.6980, 4.6980, 4.6980, 4.6980],
# CHECK:                [3.6407, 3.6407, 3.6407, 3.6407]{{\]}})
# CHECK: torch.mlir
# CHECK:   {{\[}}[4.477828  4.477828  4.477828  4.477828 ]
# CHECK:         [5.7501717 5.7501717 5.7501717 5.7501717]
# CHECK:         [4.697952  4.697952  4.697952  4.697952 ]
# CHECK:         [3.640687  3.640687  3.640687  3.640687 ]{{\]}}
#
def test_sparse_gcn():
    class GraphConv(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(GraphConv, self).__init__()
            self.kernel = nn.Parameter(torch.Tensor(input_dim, output_dim))
            nn.init.ones_(self.kernel)
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.ones_(self.bias)

        def forward(self, inp, adj_mat):
            # Input matrix times weight matrix.
            support = torch.mm(inp, self.kernel)
            # Sparse adjacency matrix times support matrix.
            output = torch.spmm(adj_mat, support)
            # Add bias.
            output = output + self.bias
            return output

    net = GraphConv(4, 4)

    # Get a random (but reproducible) matrices.
    torch.manual_seed(0)
    inp = torch.rand(4, 4)
    adj_mat = torch.rand(4, 4).to_sparse()
    m = export_and_import(net, inp, adj_mat)
    print(m)

    # Run it with PyTorch torch.sparse and with TORCH-MLIR sparse_jit.
    # Set to inference mode to avoid autograd component in result.
    with torch.no_grad():
        res1 = net(inp, adj_mat)
        res2 = sparse_jit(net, inp, adj_mat)
        print("torch.sparse")
        print(res1)
        print("torch.mlir")
        print(res2)
