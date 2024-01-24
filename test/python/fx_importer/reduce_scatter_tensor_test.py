# Copyright 2023 Advanced Micro Devices, Inc
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch
import torch.multiprocessing as mp 
import torch.distributed as dist 
import torch.export
import torch.nn as nn
import torch.distributed._functional_collectives as funcol
import os 

from typing import Optional
from torch_mlir.extras.fx_importer import FxImporter
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch.distributed import init_process_group

def export_and_import(
    f,
    *args,
    fx_importer: Optional[FxImporter] = None,
    constraints: Optional[torch.export.Constraint] = None,
    **kwargs,
):
    context = ir.Context()
    torch_d.register_dialect(context)

    if fx_importer is None:
        fx_importer = FxImporter(context=context)
    prog = torch.export.export(f, args, kwargs, constraints=constraints)
    fx_importer.import_frozen_exported_program(prog)
    return fx_importer.module_op

def test_import_frozen_exported_program(rank):
    class Reduce_Scatter_Tensor(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            out = funcol.reduce_scatter_tensor(x, "sum", 0, [0, 1, 2, 3])
            return out
    
    x = torch.arange(4, dtype=torch.float32) + rank * 4
    m = export_and_import(Reduce_Scatter_Tensor(), x)
    if (rank == 0):
        f = test_import_frozen_exported_program
        print(f"{f.__name__}")
        print("-"*len(f.__name__))
        print(m)
        print()

# CHECK: module
# CHECK-LABEL:   func.func @main(
# CHECK-SAME:                    %[[VAL_0:.*]]: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1],f32> {
# CHECK:           %[[VAL_1:.*]] = torch.constant.str "sum"
# CHECK:           %[[VAL_2:.*]] = torch.constant.str ""
# CHECK:           %[[VAL_3:.*]] = torch.constant.int 0
# CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
# CHECK:           %[[VAL_5:.*]] = torch.constant.int 2
# CHECK:           %[[VAL_6:.*]] = torch.constant.int 3
# CHECK:           %[[VAL_7:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
# CHECK:           %[[VAL_8:.*]] = torch.constant.int 4
# CHECK:           %[[VAL_9:.*]] = torch.c10d_functional.reduce_scatter_tensor %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_7]], %[[VAL_8]] : !torch.vtensor<[4],f32>, !torch.str, !torch.str, !torch.list<int>, !torch.int -> !torch.vtensor<[1],f32>
# CHECK:           %[[VAL_10:.*]] = torch.c10d_functional.wait_tensor %[[VAL_9]] : !torch.vtensor<[1],f32> -> !torch.vtensor<[1],f32>
# CHECK:           return %[[VAL_10]] : !torch.vtensor<[1],f32>
# CHECK:         }

def setup(world_size, rank, port="4991", addr="localhost"):
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.barrier()
    dist.destroy_process_group()

def run(
    rank,
    world_size,
):
    setup(world_size, rank)
    test_import_frozen_exported_program(rank)
    cleanup()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
