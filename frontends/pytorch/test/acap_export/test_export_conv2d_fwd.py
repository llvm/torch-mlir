# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

N = 3
Cin = 16
Cout = 4
w = 10
h = 10

model = torch.nn.Conv2d(Cin, Cout, (3,3))
ref_model = torch.nn.Conv2d(Cin, Cout, (3,3))

ref_model.weight.data = model.weight.clone()
ref_model.bias.data = model.bias.clone()

softmax = torch.nn.LogSoftmax(dim=1)
loss = torch.nn.NLLLoss()

tensor = torch.randn(N, Cin, h, w)

with mb.capture_function("conv2d_fwd", [tensor]) as f:
  result = model(tensor)
  f.returns([result])

# Generated with mlir/utils/generate-test-checks.py
# This is very deterministic and a change test is appropriate.
# CHECK-LABEL:   func @conv2d_fwd(
# CHECK-SAME:                     %[[VAL_0:.*]]: !numpy.ndarray<[3,16,10,10]:f32>) -> !numpy.ndarray<[3,4,8,8]:f32> {
# CHECK:           %[[VAL_1:.*]] = constant opaque<"", "0xDEADBEEF"> : tensor<4x16x3x3xf32>
# CHECK:           %[[VAL_2:.*]] = constant opaque<"", "0xDEADBEEF"> : tensor<4xf32>
# CHECK:           %[[VAL_3:.*]] = constant 1 : i64
# CHECK:           %[[VAL_4:.*]] = constant 1 : i64
# CHECK:           %[[VAL_5:.*]] = basicpy.build_list %[[VAL_3]], %[[VAL_4]] : (i64, i64) -> !basicpy.ListType
# CHECK:           %[[VAL_6:.*]] = constant 0 : i64
# CHECK:           %[[VAL_7:.*]] = constant 0 : i64
# CHECK:           %[[VAL_8:.*]] = basicpy.build_list %[[VAL_6]], %[[VAL_7]] : (i64, i64) -> !basicpy.ListType
# CHECK:           %[[VAL_9:.*]] = constant 1 : i64
# CHECK:           %[[VAL_10:.*]] = constant 1 : i64
# CHECK:           %[[VAL_11:.*]] = basicpy.build_list %[[VAL_9]], %[[VAL_10]] : (i64, i64) -> !basicpy.ListType
# CHECK:           %[[VAL_12:.*]] = constant false
# CHECK:           %[[VAL_13:.*]] = constant 0 : i64
# CHECK:           %[[VAL_14:.*]] = constant 0 : i64
# CHECK:           %[[VAL_15:.*]] = basicpy.build_list %[[VAL_13]], %[[VAL_14]] : (i64, i64) -> !basicpy.ListType
# CHECK:           %[[VAL_16:.*]] = constant 1 : i64
# CHECK:           %[[VAL_17:.*]] = numpy.create_array_from_tensor %[[VAL_1]] : (tensor<4x16x3x3xf32>) -> !numpy.ndarray<[4,16,3,3]:f32>
# CHECK:           %[[VAL_18:.*]] = numpy.create_array_from_tensor %[[VAL_2]] : (tensor<4xf32>) -> !numpy.ndarray<[4]:f32>
# CHECK:           %[[VAL_19:.*]] = torch.kernel_call "aten::convolution" %[[VAL_0]], %[[VAL_17]], %[[VAL_18]], %[[VAL_5]], %[[VAL_8]], %[[VAL_11]], %[[VAL_12]], %[[VAL_15]], %[[VAL_16]] : (!numpy.ndarray<[3,16,10,10]:f32>, !numpy.ndarray<[4,16,3,3]:f32>, !numpy.ndarray<[4]:f32>, !basicpy.ListType, !basicpy.ListType, !basicpy.ListType, i1, !basicpy.ListType, i64) -> !numpy.ndarray<[3,4,8,8]:f32> {sigArgTypes = ["Tensor", "Tensor", "Tensor?", "int[]", "int[]", "int[]", "bool", "int[]", "int"], sigIsMutable = false, sigIsVararg = false, sigIsVarret = false, sigRetTypes = ["Tensor"]}
# CHECK:           return %[[VAL_19]] : !numpy.ndarray<[3,4,8,8]:f32>
# CHECK:         }
mb.module.operation.print(large_elements_limit=2)
