# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import npcomp.frontends.pytorch as torch_mlir
import npcomp.frontends.pytorch.test as test

# RUN: %PYTHON %s | FileCheck %s

dev = torch_mlir.mlir_device()

x = torch.rand((3,64,8,8), device=dev)
y = x*x
print (y.stride())

dim = [64,24,24]
dim = [4,4,4]
N = 2;
count = dim[0]*dim[1]*dim[2]
sizes = (N,dim[0],dim[1],dim[2])
strides = (1,dim[1]*dim[2],dim[2],1)
print(count)
t0 = torch.randn((N,count), device=dev)
t0_like = torch.randn((N,count))


t1 = t0.as_strided(sizes, strides)
t1_ref = t0.to('cpu').as_strided(sizes, strides)
t1_like = t0_like.as_strided(sizes, strides)

t1_ref = t1_ref.clone()

# check that the IR has recorded the
# stride properly before invoking JIT
# CHECK: PASS! stride check
test.compare_eq(t1.stride(), t1_like.stride(), "stride")

# CHECK: PASS! as_stride check
test.compare(t1_ref, t1, "as_stride")

# CHECK: PASS! as_stride stride check
test.compare_eq(t1_ref.stride(), t1.to("cpu").stride(), "as_stride stride")
