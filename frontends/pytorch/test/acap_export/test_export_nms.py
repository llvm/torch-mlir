# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torchvision
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

boxes = torch.rand(50, 4)
scores = torch.rand(50)

with mb.capture_function("nms", [boxes, scores]) as f:
    result = torch.ops.torchvision.nms(boxes, scores, 0.5)
    f.returns([result])

# CHECK-LABEL:  func @nms(%arg0: !numpy.ndarray<[50,4]:f32>, %arg1: !numpy.ndarray<[50]:f32>) -> !numpy.ndarray<[50]:i64> {
# CHECK:           %[[VAL_0:.*]] = constant 5.000000e-01 : f64
# CHECK:           %[[VAL_1:.*]] = torch.kernel_call "torchvision::nms" %arg0, %arg1, %[[VAL_0]] : (!numpy.ndarray<[50,4]:f32>, !numpy.ndarray<[50]:f32>, f64) -> !numpy.ndarray<[50]:i64> {sigArgTypes = ["Tensor", "Tensor", "float"], sigIsMutable = false, sigIsVararg = false, sigIsVarret = false, sigRetTypes = ["Tensor"]}
# CHECK:           return %[[VAL_1]] : !numpy.ndarray<[50]:i64>
# CHECK:        }
mb.module.operation.print(large_elements_limit=2)
