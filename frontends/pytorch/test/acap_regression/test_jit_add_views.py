# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import npcomp.frontends.pytorch as torch_mlir
import npcomp.frontends.pytorch.test as test

# RUN: %PYTHON %s | FileCheck %s

dev = torch_mlir.mlir_device()
t0 = torch.randn((4,16,4), device=dev)
t1 = torch.randn((4,16,4), device=dev)

t3 = torch.randn((4,64), device=dev)
t4 = torch.randn((4,64), device=dev)

t2 = t0 + t1
t5 = t3 + t4

t6 = t5.view((4,4,4,4))
t7 = t2.view((4,4,4,4))

t8 = t6 + t7

t0_cpu = t0.to('cpu')
t1_cpu = t1.to('cpu')

# CHECK: PASS! add_views_0 check
test.compare(t2, t0_cpu + t1_cpu, "add_views_0")

t3_cpu = t3.to('cpu')
t4_cpu = t4.to('cpu')

# CHECK: PASS! add_views_1 check
test.compare(t5, t3_cpu + t4_cpu, "add_views_1")

t6_cpu = t6.to('cpu')
t7_cpu = t7.to('cpu')

# CHECK: PASS! add_views_2 check
test.compare(t8, t6_cpu + t7_cpu, "add_views_2")
