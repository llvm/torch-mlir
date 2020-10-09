# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import npcomp.frontends.pytorch as torch_mlir
import npcomp.frontends.pytorch.test as test

# RUN: %PYTHON %s | FileCheck %s

dev = torch_mlir.mlir_device()
t0 = torch.randn((1,2,3,4), device=dev)
t1 = torch.randn((1,2,3,4), device=dev)
t2 = torch.randn((1,2,3,4), device=dev)

t3 = t0 + t1 + t2

#
# Check the result tensor against the CPU
#
t0_cpu = t0.to('cpu')
t1_cpu = t1.to('cpu')
t2_cpu = t2.to('cpu')
t3_cpu = t3.to('cpu')

print (t0_cpu, " +\n", t1_cpu, " +\n", t2_cpu, " =\n", t3_cpu)

# CHECK: PASS!
test.compare(t3, t0_cpu + t1_cpu + t2_cpu, "add3")
