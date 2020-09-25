# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import npcomp.frontends.pytorch as torch_mlir
import npcomp.frontends.pytorch.test as test

# RUN: python %s | FileCheck %s

dev = torch_mlir.mlir_device()

t0 = torch.randn((3,13), device=dev)
t1 = torch.randn((13,5), device=dev)
print(t0.to('cpu'), t1.to('cpu'))
print(torch.mm(t0.to('cpu'), t1.to('cpu')))

t2 = torch.mm(t0, t1)

#
# Check the result tensor against the CPU
#
t0_cpu = t0.to('cpu')
t1_cpu = t1.to('cpu')
t2_cpu = t2.to('cpu')

print (t0_cpu, " *\n", t1_cpu, " =\n", t2_cpu)

ref_tensor = torch.mm(t0_cpu, t1_cpu)
# CHECK: PASS! mm check
test.compare(t2, ref_tensor, "mm")

