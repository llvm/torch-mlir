# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import npcomp.frontends.pytorch as torch_mlir

# RUN: python %s | FileCheck %s

dev = torch_mlir.mlir_device()

model = torch.nn.Conv2d(2,16,7,stride=[2,2], padding=[3,3], dilation=1, groups=1, bias=True).to(dev)

tensor = torch.randn((1,2,128,128), device=dev)
result = model(tensor)

mlir = torch_mlir.get_mlir( result )
report = torch_mlir.op_report(mlir)

# CHECK-LABEL:   "L0-convolution_overrideable-0"
#   CHECK-NEXT:     "activation_in": 32768
#   CHECK-NEXT:     "activation_out": 65536
#   CHECK-NEXT:     "ops:+": 65536
#   CHECK-NEXT:     "ops:MAC": 6422528
#   CHECK-NEXT:     "parameters_in": 1584
#   CHECK-NEXT:     "reads": 34352
#   CHECK-NEXT:     "writes": 65536
for k,v in report.items():
    print("\"{}\"".format(k))
    for k,v in v.items():
        print("\"{}\": {}".format(k,v))
