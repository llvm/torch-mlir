// RUN: torch-mlir-opt <%s -convert-torch-to-mhlo -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.tanh$basic(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = mhlo.tanh %[[VAL_1]] : tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.tanh$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.tanh %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.log$basic(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = mhlo.log %[[VAL_1]] : tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.log$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.log %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.exp$basic(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = mhlo.exponential %[[VAL_1]] : tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.from_builtin_tensor %[[VAL_2]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_3]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.exp$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.exp %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.clone$basic(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = "mhlo.copy"(%[[VAL_1]]) : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.clone$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %none = torch.constant.none
  %0 = torch.aten.clone %arg0, %none : !torch.vtensor<[?,?],f32>, !torch.none -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.addscalar$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 9
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = mhlo.constant dense<9.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_5:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_4]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<4x64xf32>
// CHECK:           %[[VAL_6:.*]] = mhlo.add %[[VAL_1]], %[[VAL_5]] : tensor<4x64xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.addscalar$basic(%arg0: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %int9 = torch.constant.int 9
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Scalar %arg0, %int9, %int1 : !torch.vtensor<[4,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.addscalar$basic_dynamic(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %int9 = torch.constant.int 9
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<9.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = shape.shape_of %[[VAL_1]] : tensor<?x?xf32> -> tensor<2xindex>
// CHECK:           %[[VAL_4:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[VAL_2]], %[[VAL_3]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = mhlo.add %[[VAL_1]], %[[VAL_4]] : tensor<?x?xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.addscalar$basic_dynamic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int9 = torch.constant.int 9
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Scalar %arg0, %int9, %int1 : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.addtensor$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_4:.*]] = mhlo.add %[[VAL_2]], %[[VAL_3]] : tensor<4x64xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.addtensor$basic(%arg0: !torch.vtensor<[4,64],f32>, %arg1: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[4,64],f32>, !torch.vtensor<[4,64],f32>, !torch.int -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.addtensor$promote(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !torch.vtensor<[4,64],si32>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: !torch.vtensor<[4,64],si64>) -> !torch.vtensor<[4,64],si64> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],si32> -> tensor<4x64xi32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,64],si64> -> tensor<4x64xi64>
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = mhlo.convert(%[[VAL_2]]) : (tensor<4x64xi32>) -> tensor<4x64xi64>
// CHECK:           %[[VAL_6:.*]] = mhlo.add %[[VAL_5]], %[[VAL_3]] : tensor<4x64xi64>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<4x64xi64> -> !torch.vtensor<[4,64],si64>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[4,64],si64>
func.func @torch.aten.addtensor$promote(%arg0: !torch.vtensor<[4,64],si32>, %arg1: !torch.vtensor<[4,64],si64>) -> !torch.vtensor<[4,64],si64> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[4,64],si32>, !torch.vtensor<[4,64],si64>, !torch.int -> !torch.vtensor<[4,64],si64>
  return %0 : !torch.vtensor<[4,64],si64>
}

// -----

// CHECK-LABEL:    func.func @torch.aten.addtensor$bcast(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !torch.vtensor<[64],f32>,
// CHECK-SAME:                                      %[[VAL_1:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:            %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[64],f32> -> tensor<64xf32>
// CHECK:            %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:            %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:            %[[VAL_5:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_2]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<4x64xf32>
// CHECK:            %[[VAL_6:.*]] = mhlo.add %[[VAL_5]], %[[VAL_3]] : tensor<4x64xf32>
// CHECK:            %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:            return %[[VAL_7]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.addtensor$bcast(%arg0: !torch.vtensor<[64],f32>, %arg1: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[64],f32>, !torch.vtensor<[4,64],f32>, !torch.int -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.addtensor$basic_dynamic(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>, 
// CHECK-SAME:                                                  %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = shape.shape_of %[[VAL_2]] : tensor<?x?xf32> -> tensor<2xindex>
// CHECK:           %[[VAL_5:.*]] = shape.shape_of %[[VAL_3]] : tensor<?x?xf32> -> tensor<2xindex>
// CHECK:           %[[VAL_6:.*]] = shape.cstr_broadcastable %[[VAL_4]], %[[VAL_5]] : tensor<2xindex>, tensor<2xindex>
// CHECK:           %[[VAL_7:.*]]:3 = shape.assuming %[[VAL_6]] -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<2xindex>) {
// CHECK:             %[[VAL_10:.*]] = shape.broadcast %[[VAL_4]], %[[VAL_5]] : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
// CHECK:             %[[VAL_11:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[VAL_2]], %[[VAL_10]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:             %[[VAL_12:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[VAL_3]], %[[VAL_10]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:             shape.assuming_yield %[[VAL_11]], %[[VAL_12]], %[[VAL_10]] : tensor<?x?xf32>, tensor<?x?xf32>, tensor<2xindex>
// CHECK:           }
// CHECK:           %[[VAL_8:.*]] = mhlo.add %[[VAL_7]]#0, %[[VAL_7]]#1 : tensor<?x?xf32>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_9]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.addtensor$basic_dynamic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.addtensor$alpha(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_5:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_5]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<4x64xf32>
// CHECK:           %[[VAL_7:.*]] = mhlo.multiply %[[VAL_3]], %[[VAL_6]] : tensor<4x64xf32>
// CHECK:           %[[VAL_8:.*]] = mhlo.add %[[VAL_2]], %[[VAL_7]] : tensor<4x64xf32>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:           return %[[VAL_9]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.addtensor$alpha(%arg0: !torch.vtensor<[4,64],f32>, %arg1: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %int2 = torch.constant.int 2
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int2 : !torch.vtensor<[4,64],f32>, !torch.vtensor<[4,64],f32>, !torch.int -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.mulscalar$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 9
// CHECK:           %[[VAL_3:.*]] = mhlo.constant dense<9.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_4:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_3]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<4x64xf32>
// CHECK:           %[[VAL_5:.*]] = mhlo.multiply %[[VAL_1]], %[[VAL_4]] : tensor<4x64xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.mulscalar$basic(%arg0: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %int9 = torch.constant.int 9
  %0 = torch.aten.mul.Scalar %arg0, %int9 : !torch.vtensor<[4,64],f32>, !torch.int -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.multensor$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_4:.*]] = mhlo.multiply %[[VAL_2]], %[[VAL_3]] : tensor<4x64xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.multensor$basic(%arg0: !torch.vtensor<[4,64],f32>, %arg1: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[4,64],f32>, !torch.vtensor<[4,64],f32> -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.multensor$bcast(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[8,4,64],f32>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: !torch.vtensor<[8,1,64],f32>) -> !torch.vtensor<[8,4,64],f32> {
// CHECK:          %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[8,4,64],f32> -> tensor<8x4x64xf32>
// CHECK:          %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[8,1,64],f32> -> tensor<8x1x64xf32>
// CHECK:          %[[VAL_4:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_3]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<8x1x64xf32>) -> tensor<8x4x64xf32>
// CHECK:          %[[VAL_5:.*]] = mhlo.multiply %[[VAL_2]], %[[VAL_4]] : tensor<8x4x64xf32>
// CHECK:          %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<8x4x64xf32> -> !torch.vtensor<[8,4,64],f32>
// CHECK:          return %[[VAL_6]] : !torch.vtensor<[8,4,64],f32>
func.func @torch.aten.multensor$bcast(%arg0: !torch.vtensor<[8,4,64],f32>, %arg1: !torch.vtensor<[8,1,64],f32>) -> !torch.vtensor<[8,4,64],f32> {
  %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[8,4,64],f32>, !torch.vtensor<[8,1,64],f32> -> !torch.vtensor<[8,4,64],f32>
  return %0 : !torch.vtensor<[8,4,64],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.subscalar$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 9
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = mhlo.constant dense<9.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_5:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_4]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<4x64xf32>
// CHECK:           %[[VAL_6:.*]] = mhlo.subtract %[[VAL_1]], %[[VAL_5]] : tensor<4x64xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.subscalar$basic(%arg0: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %int9 = torch.constant.int 9
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Scalar %arg0, %int9, %int1 : !torch.vtensor<[4,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.subtensor$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_4:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_3]] : tensor<4x64xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.subtensor$basic(%arg0: !torch.vtensor<[4,64],f32>, %arg1: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[4,64],f32>, !torch.vtensor<[4,64],f32>, !torch.int -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.subtensor$promote(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !torch.vtensor<[4,64],si32>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: !torch.vtensor<[4,64],si64>) -> !torch.vtensor<[4,64],si64> {
// CHECK:          %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],si32> -> tensor<4x64xi32>
// CHECK:          %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,64],si64> -> tensor<4x64xi64>
// CHECK:          %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:          %[[VAL_5:.*]] = mhlo.convert(%[[VAL_2]]) : (tensor<4x64xi32>) -> tensor<4x64xi64>
// CHECK:          %[[VAL_6:.*]] = mhlo.subtract %[[VAL_5]], %[[VAL_3]] : tensor<4x64xi64>
// CHECK:          %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<4x64xi64> -> !torch.vtensor<[4,64],si64>
// CHECK:          return %[[VAL_7]] : !torch.vtensor<[4,64],si64>
func.func @torch.aten.subtensor$promote(%arg0: !torch.vtensor<[4,64],si32>, %arg1: !torch.vtensor<[4,64],si64>) -> !torch.vtensor<[4,64],si64> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[4,64],si32>, !torch.vtensor<[4,64],si64>, !torch.int -> !torch.vtensor<[4,64],si64>
  return %0 : !torch.vtensor<[4,64],si64>
}

// -----

// CHECK-LABEL:    func.func @torch.aten.subtensor$bcast(
// CHECK-SAME:                                           %[[VAL_0:.*]]: !torch.vtensor<[64],f32>,
// CHECK-SAME:                                           %[[VAL_1:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:            %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[64],f32> -> tensor<64xf32>
// CHECK:            %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:            %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:            %[[VAL_5:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_2]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<4x64xf32>
// CHECK:            %[[VAL_6:.*]] = mhlo.subtract %[[VAL_5]], %[[VAL_3]] : tensor<4x64xf32>
// CHECK:            %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:            return %[[VAL_7]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.subtensor$bcast(%arg0: !torch.vtensor<[64],f32>, %arg1: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[64],f32>, !torch.vtensor<[4,64],f32>, !torch.int -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.subtensor$alpha(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_5:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_5]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<4x64xf32>
// CHECK:           %[[VAL_7:.*]] = mhlo.multiply %[[VAL_3]], %[[VAL_6]] : tensor<4x64xf32>
// CHECK:           %[[VAL_8:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_7]] : tensor<4x64xf32>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:           return %[[VAL_9]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.subtensor$alpha(%arg0: !torch.vtensor<[4,64],f32>, %arg1: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %int2 = torch.constant.int 2
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %int2 : !torch.vtensor<[4,64],f32>, !torch.vtensor<[4,64],f32>, !torch.int -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.divscalar$basic(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 9
// CHECK:           %[[VAL_3:.*]] = mhlo.constant dense<9.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_4:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_3]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<4x64xf32>
// CHECK:           %[[VAL_5:.*]] = mhlo.divide %[[VAL_1]], %[[VAL_4]] : tensor<4x64xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.divscalar$basic(%arg0: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %int9 = torch.constant.int 9
  %0 = torch.aten.div.Scalar %arg0, %int9 : !torch.vtensor<[4,64],f32>, !torch.int -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.divscalar$dynamic(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %int9 = torch.constant.int 9
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<9.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = shape.shape_of %[[VAL_1]] : tensor<?x?xf32> -> tensor<2xindex>
// CHECK:           %[[VAL_4:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[VAL_2]], %[[VAL_3]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = mhlo.divide %[[VAL_1]], %[[VAL_4]] : tensor<?x?xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.divscalar$dynamic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int9 = torch.constant.int 9
  %0 = torch.aten.div.Scalar %arg0, %int9 : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.divtensor$basic(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_4:.*]] = mhlo.divide %[[VAL_2]], %[[VAL_3]] : tensor<4x64xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.divtensor$basic(%arg0: !torch.vtensor<[4,64],f32>, %arg1: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %0 = torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[4,64],f32>, !torch.vtensor<[4,64],f32> -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.divtensor$dynamic(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>, 
// CHECK-SAME:                                          %[[VAL_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = shape.shape_of %[[VAL_1:.*]] : tensor<?x?xf32> -> tensor<2xindex>
// CHECK:           %[[VAL_5:.*]] = shape.shape_of %[[VAL_3:.*]] : tensor<?x?xf32> -> tensor<2xindex>
// CHECK:           %[[VAL_6:.*]] = shape.cstr_broadcastable %[[VAL_4]], %[[VAL_5]] : tensor<2xindex>, tensor<2xindex>
// CHECK:           %[[VAL_7:.*]]:3 = shape.assuming %[[VAL_6:.*]] -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<2xindex>) {
// CHECK:             %[[VAL_9:.*]] = shape.broadcast %[[VAL_4]], %[[VAL_5]] : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
// CHECK:             %[[VAL_10:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[VAL_2]], %[[VAL_9]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:             %[[VAL_11:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[VAL_3]], %[[VAL_9]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:             shape.assuming_yield %[[VAL_10]], %[[VAL_11]], %[[VAL_9]] : tensor<?x?xf32>, tensor<?x?xf32>, tensor<2xindex>
// CHECK:           }
// CHECK:           %[[VAL_8:.*]] = mhlo.divide %[[VAL_7]]#0, %[[VAL_7]]#1 : tensor<?x?xf32>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_9]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.divtensor$dynamic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.divtensor$bcast(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !torch.vtensor<[8,4,64],f32>,
// CHECK-SAME:                                         %[[VAL_1:.*]]: !torch.vtensor<[8,1,64],f32>) -> !torch.vtensor<[8,4,64],f32> {
// CHECK:          %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[8,4,64],f32> -> tensor<8x4x64xf32>
// CHECK:          %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[8,1,64],f32> -> tensor<8x1x64xf32>
// CHECK:          %[[VAL_4:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_3]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<8x1x64xf32>) -> tensor<8x4x64xf32>
// CHECK:          %[[VAL_5:.*]] = mhlo.divide %[[VAL_2]], %[[VAL_4]] : tensor<8x4x64xf32>
// CHECK:          %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<8x4x64xf32> -> !torch.vtensor<[8,4,64],f32>
// CHECK:          return %[[VAL_6]] : !torch.vtensor<[8,4,64],f32>
func.func @torch.aten.divtensor$bcast(%arg0: !torch.vtensor<[8,4,64],f32>, %arg1: !torch.vtensor<[8,1,64],f32>) -> !torch.vtensor<[8,4,64],f32> {
  %0 = torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[8,4,64],f32>, !torch.vtensor<[8,1,64],f32> -> !torch.vtensor<[8,4,64],f32>
  return %0 : !torch.vtensor<[8,4,64],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.vtensor.literal$basic() -> !torch.vtensor<[],f32> {
// CHECK:           %[[VAL_0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:           return %[[VAL_1]] : !torch.vtensor<[],f32>
func.func @torch.vtensor.literal$basic() -> !torch.vtensor<[],f32> {
  %0 = torch.vtensor.literal(dense<0.0> : tensor<f32>) : !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.vtensor.literal$signed() -> !torch.vtensor<[2],si64> {
// CHECK:           %[[VAL_0:.*]] = mhlo.constant dense<1> : tensor<2xi64>
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<2xi64> -> !torch.vtensor<[2],si64>
// CHECK:           return %[[VAL_1]] : !torch.vtensor<[2],si64>
func.func @torch.vtensor.literal$signed() -> !torch.vtensor<[2],si64> {
  %0 = torch.vtensor.literal(dense<1> : tensor<2xsi64>) : !torch.vtensor<[2],si64>
  return %0 : !torch.vtensor<[2],si64>
}

// -----

// CHECK-LABEL: func.func @torch.aten.gt.scalar(
// CHECK-SAME:                             %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],i1> {
// CHECK:         %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:         %[[VAL_2:.*]] = torch.constant.int 3
// CHECK:         %[[VAL_3:.*]] = mhlo.constant dense<3.000000e+00> : tensor<f32>
// CHECK:         %[[VAL_4:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_3]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<4x64xf32>
// CHECK:         %[[VAL_5:.*]] = "mhlo.compare"(%[[VAL_1]], %[[VAL_4]]) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<4x64xf32>, tensor<4x64xf32>) -> tensor<4x64xi1>
// CHECK:         %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<4x64xi1> -> !torch.vtensor<[4,64],i1>
// CHECK:         return %[[VAL_6]] : !torch.vtensor<[4,64],i1>
func.func @torch.aten.gt.scalar(%arg0: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],i1> {
  %int3 = torch.constant.int 3
  %0 = torch.aten.gt.Scalar %arg0, %int3 : !torch.vtensor<[4,64],f32>, !torch.int -> !torch.vtensor<[4,64],i1>
  return %0 : !torch.vtensor<[4,64],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.gt.scalar$dynamic(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],i1> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %int3 = torch.constant.int 3
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<3.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = shape.shape_of %[[VAL_1]] : tensor<?x?xf32> -> tensor<2xindex>
// CHECK:           %[[VAL_4:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[VAL_2]], %[[VAL_3]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = "mhlo.compare"(%[[VAL_1]], %[[VAL_4]]) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<?x?xi1> -> !torch.vtensor<[?,?],i1>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[?,?],i1>
func.func @torch.aten.gt.scalar$dynamic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],i1> {
  %int3 = torch.constant.int 3
  %0 = torch.aten.gt.Scalar %arg0, %int3 : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// -----

// CHECK-LABEL:    func.func @torch.aten.gt.tensor(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>, 
// CHECK-SAME:                                %[[VAL_1:.*]]: !torch.vtensor<[64],f32>) -> !torch.vtensor<[4,64],i1> {
// CHECK:            %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:            %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[64],f32> -> tensor<64xf32>
// CHECK:            %[[VAL_4:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_3]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<4x64xf32>
// CHECK:            %[[VAL_5:.*]] = "mhlo.compare"(%[[VAL_2]], %[[VAL_4]]) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<4x64xf32>, tensor<4x64xf32>) -> tensor<4x64xi1>
// CHECK:            %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<4x64xi1> -> !torch.vtensor<[4,64],i1>
// CHECK:            return %[[VAL_6]] : !torch.vtensor<[4,64],i1>
func.func @torch.aten.gt.tensor(%arg0: !torch.vtensor<[4,64],f32>, %arg1: !torch.vtensor<[64],f32>) -> !torch.vtensor<[4,64],i1> {
  %0 = torch.aten.gt.Tensor %arg0, %arg1 : !torch.vtensor<[4,64],f32>, !torch.vtensor<[64],f32> -> !torch.vtensor<[4,64],i1>
  return %0 : !torch.vtensor<[4,64],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.gt.tensor$dynamic(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>, 
// CHECK-SAME:                                            %[[VAL_1:.*]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?],i1> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[?],f32> -> tensor<?xf32>
// CHECK:           %[[VAL_4:.*]] = shape.shape_of %[[VAL_2]] : tensor<?x?xf32> -> tensor<2xindex>
// CHECK:           %[[VAL_5:.*]] = shape.shape_of %[[VAL_3]] : tensor<?xf32> -> tensor<1xindex>
// CHECK:           %[[VAL_6:.*]] = shape.cstr_broadcastable %[[VAL_4]], %[[VAL_5]] : tensor<2xindex>, tensor<1xindex>
// CHECK:           %[[VAL_7:.*]]:3 = shape.assuming %[[VAL_6]] -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<2xindex>) {
// CHECK:             %[[VAL_10:.*]] = shape.broadcast %[[VAL_4]], %[[VAL_5]] : tensor<2xindex>, tensor<1xindex> -> tensor<2xindex>
// CHECK:             %[[VAL_11:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[VAL_2]], %[[VAL_10]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:             %[[VAL_12:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[VAL_3]], %[[VAL_10]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:             shape.assuming_yield %[[VAL_11]], %[[VAL_12]], %[[VAL_10]] : tensor<?x?xf32>, tensor<?x?xf32>, tensor<2xindex>
// CHECK:           }
// CHECK:           %[[VAL_8:.*]] = "mhlo.compare"(%[[VAL_7]]#0, %[[VAL_7]]#1) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<?x?xi1> -> !torch.vtensor<[?,?],i1>
// CHECK:           return %[[VAL_9]] : !torch.vtensor<[?,?],i1>
func.func @torch.aten.gt.tensor$dynamic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?],i1> {
  %0 = torch.aten.gt.Tensor %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?],f32> -> !torch.vtensor<[?,?],i1>
  return %0 : !torch.vtensor<[?,?],i1>
}

// -----

// CHECK-LABEL:    func.func @torch.aten.lt.tensor(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>, 
// CHECK-SAME:                                %[[VAL_1:.*]]: !torch.vtensor<[64],f32>) -> !torch.vtensor<[4,64],i1> {
// CHECK:            %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:            %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[64],f32> -> tensor<64xf32>
// CHECK:            %[[VAL_4:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_3]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<4x64xf32>
// CHECK:            %[[VAL_5:.*]] = "mhlo.compare"(%[[VAL_2]], %[[VAL_4]]) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<4x64xf32>, tensor<4x64xf32>) -> tensor<4x64xi1>
// CHECK:            %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<4x64xi1> -> !torch.vtensor<[4,64],i1>
// CHECK:            return %[[VAL_6]] : !torch.vtensor<[4,64],i1>
func.func @torch.aten.lt.tensor(%arg0: !torch.vtensor<[4,64],f32>, %arg1: !torch.vtensor<[64],f32>) -> !torch.vtensor<[4,64],i1> {
  %0 = torch.aten.lt.Tensor %arg0, %arg1 : !torch.vtensor<[4,64],f32>, !torch.vtensor<[64],f32> -> !torch.vtensor<[4,64],i1>
  return %0 : !torch.vtensor<[4,64],i1>
}

// -----

// CHECK-LABEL:    func.func @torch.aten.eq.tensor(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>, 
// CHECK-SAME:                                        %[[VAL_1:.*]]: !torch.vtensor<[64],f32>) -> !torch.vtensor<[4,64],i1> {
// CHECK:            %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:            %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[64],f32> -> tensor<64xf32>
// CHECK:            %[[VAL_4:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_3]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<4x64xf32>
// CHECK:            %[[VAL_5:.*]] = "mhlo.compare"(%[[VAL_2]], %[[VAL_4]]) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<4x64xf32>, tensor<4x64xf32>) -> tensor<4x64xi1>
// CHECK:            %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<4x64xi1> -> !torch.vtensor<[4,64],i1>
// CHECK:            return %[[VAL_6]] : !torch.vtensor<[4,64],i1>
func.func @torch.aten.eq.tensor(%arg0: !torch.vtensor<[4,64],f32>, %arg1: !torch.vtensor<[64],f32>) -> !torch.vtensor<[4,64],i1> {
  %0 = torch.aten.eq.Tensor %arg0, %arg1 : !torch.vtensor<[4,64],f32>, !torch.vtensor<[64],f32> -> !torch.vtensor<[4,64],i1>
  return %0 : !torch.vtensor<[4,64],i1>
}

// -----

// CHECK-LABEL:    func.func @torch.aten.ne.tensor(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>, 
// CHECK-SAME:                                %[[VAL_1:.*]]: !torch.vtensor<[64],f32>) -> !torch.vtensor<[4,64],i1> {
// CHECK:            %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:            %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[64],f32> -> tensor<64xf32>
// CHECK:            %[[VAL_4:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_3]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<4x64xf32>
// CHECK:            %[[VAL_5:.*]] = "mhlo.compare"(%[[VAL_2]], %[[VAL_4]]) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction NE>} : (tensor<4x64xf32>, tensor<4x64xf32>) -> tensor<4x64xi1>
// CHECK:            %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<4x64xi1> -> !torch.vtensor<[4,64],i1>
// CHECK:            return %[[VAL_6]] : !torch.vtensor<[4,64],i1>
func.func @torch.aten.ne.tensor(%arg0: !torch.vtensor<[4,64],f32>, %arg1: !torch.vtensor<[64],f32>) -> !torch.vtensor<[4,64],i1> {
  %0 = torch.aten.ne.Tensor %arg0, %arg1 : !torch.vtensor<[4,64],f32>, !torch.vtensor<[64],f32> -> !torch.vtensor<[4,64],i1>
  return %0 : !torch.vtensor<[4,64],i1>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.batch_norm(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[?,3,?,?],f32>) -> !torch.vtensor<[?,3,?,?],f32> {
// CEHCK：          %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,3,?,?],f32> -> tensor<?x3x?x?xf32>
// CEHCK：          %[[VAL_2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<3xf32>
// CEHCK：          %[[VAL_3:.*]] = mhlo.constant dense<1.000000e+00> : tensor<3xf32>
// CEHCK：          %true = torch.constant.bool true
// CEHCK：          %[[VAL_4:.*]] = mhlo.constant dense<0> : tensor<i64>
// CEHCK：          %float1.000000e-01 = torch.constant.float 1.000000e-01
// CEHCK：          %float1.000000e-05 = torch.constant.float 1.000000e-05
// CEHCK：          %int1 = torch.constant.int 1
// CEHCK：          %[[VAL_5:.*]] = mhlo.constant dense<1> : tensor<i64>
// CEHCK：          %[[VAL_6:.*]] = mhlo.add %[[VAL_4]], %[[VAL_5]] : tensor<i64>
// CEHCK：          %[[VAL_7:.*]], %batch_mean, %batch_var = "mhlo.batch_norm_training"(%[[VAL_1]], %[[VAL_3]], %[[VAL_2]]) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<?x3x?x?xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<?x3x?x?xf32>, tensor<3xf32>, tensor<3xf32>)
// CEHCK：          %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<?x3x?x?xf32> -> !torch.vtensor<[?,3,?,?],f32>
// CEHCK：          return %[[VAL_8]] : !torch.vtensor<[?,3,?,?],f32>
func.func @torch.aten.batch_norm(%arg0: !torch.vtensor<[?,3,?,?],f32>) -> !torch.vtensor<[?,3,?,?],f32> {
  %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %true = torch.constant.bool true
  %2 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
  %float1.000000e-01 = torch.constant.float 1.000000e-01
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %int1 = torch.constant.int 1
  %3 = torch.aten.add.Scalar %2, %int1, %int1 : !torch.vtensor<[],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  %4 = torch.aten.batch_norm %arg0, %1, %0, %0, %1, %true, %float1.000000e-01, %float1.000000e-05, %true : !torch.vtensor<[?,3,?,?],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[?,3,?,?],f32>
  return %4 : !torch.vtensor<[?,3,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.batch_norm$none_bias_weight(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: !torch.vtensor<[?,3,?,?],f32>) -> !torch.vtensor<[?,3,?,?],f32> {
// CEHCK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,3,?,?],f32> -> tensor<?x3x?x?xf32>
// CEHCK:           %none = torch.constant.none
// CEHCK:           %1 = mhlo.constant dense<1.000000e+00> : tensor<3xf32>
// CEHCK:           %2 = mhlo.constant dense<0.000000e+00> : tensor<3xf32>
// CEHCK:           %true = torch.constant.bool true
// CEHCK:           %[[VAL_2:.*]] = mhlo.constant dense<0> : tensor<i64>
// CEHCK:           %float1.000000e-01 = torch.constant.float 1.000000e-01
// CEHCK:           %float1.000000e-05 = torch.constant.float 1.000000e-05
// CEHCK:           %int1 = torch.constant.int 1
// CEHCK:           %[[VAL_3:.*]] = mhlo.constant dense<1> : tensor<i64>
// CEHCK:           %[[VAL_4:.*]] = mhlo.add %[[VAL_2]], %[[VAL_3]] : tensor<i64>
// CEHCK:           %[[VAL_5:.*]] = mhlo.constant dense<1.000000e+00> : tensor<3xf32>
// CEHCK:           %[[VAL_6:.*]] = mhlo.constant dense<0.000000e+00> : tensor<3xf32>
// CEHCK:           %[[VAL_7:.*]], %batch_mean, %batch_var = "mhlo.batch_norm_training"(%[[VAL_1]], %[[VAL_5]], %[[VAL_6]]) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<?x3x?x?xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<?x3x?x?xf32>, tensor<3xf32>, tensor<3xf32>)
// CEHCK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<?x3x?x?xf32> -> !torch.vtensor<[?,3,?,?],f32>
// CEHCK:           return %[[VAL_8]] : !torch.vtensor<[?,3,?,?],f32>
func.func @torch.aten.batch_norm$none_bias_weight(%arg0: !torch.vtensor<[?,3,?,?],f32>) -> !torch.vtensor<[?,3,?,?],f32> {
  %none = torch.constant.none
  %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %1 = torch.vtensor.literal(dense<0.000000e+00> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %true = torch.constant.bool true
  %2 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
  %float1.000000e-01 = torch.constant.float 1.000000e-01
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %int1 = torch.constant.int 1
  %3 = torch.aten.add.Scalar %2, %int1, %int1 : !torch.vtensor<[],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  %4 = torch.aten.batch_norm %arg0, %none, %none, %1, %0, %true, %float1.000000e-01, %float1.000000e-05, %true : !torch.vtensor<[?,3,?,?],f32>, !torch.none, !torch.none, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[?,3,?,?],f32>
  return %4 : !torch.vtensor<[?,3,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.batch_norm$inference(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[?,3,?,?],f32>) -> !torch.vtensor<[?,3,?,?],f32> {
// CEHCK：          %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,3,?,?],f32> -> tensor<?x3x?x?xf32>
// CEHCK：          %[[VAL_2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<3xf32>
// CEHCK：          %[[VAL_3:.*]] = mhlo.constant dense<1.000000e+00> : tensor<3xf32>
// CEHCK：          %true = torch.constant.bool true
// CHECK：          %false = torch.constant.bool false
// CEHCK：          %[[VAL_4:.*]] = mhlo.constant dense<0> : tensor<i64>
// CEHCK：          %float1.000000e-01 = torch.constant.float 1.000000e-01
// CEHCK：          %float1.000000e-05 = torch.constant.float 1.000000e-05
// CEHCK：          %int1 = torch.constant.int 1
// CEHCK：          %[[VAL_5:.*]] = mhlo.constant dense<1> : tensor<i64>
// CEHCK：          %[[VAL_6:.*]] = mhlo.add %[[VAL_4]], %[[VAL_5]] : tensor<i64>
// CEHCK：          %[[VAL_7:.*]], %batch_mean, %batch_var = "mhlo.batch_norm_training"(%[[VAL_1]], %[[VAL_3]], %[[VAL_2]], %[[VAL_2]],  %[[VAL_3]]) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<?x3x?x?xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<?x3x?x?xf32>, tensor<3xf32>, tensor<3xf32>)
// CEHCK：          %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<?x3x?x?xf32> -> !torch.vtensor<[?,3,?,?],f32>
// CEHCK：          return %[[VAL_8]] : !torch.vtensor<[?,3,?,?],f32>
func.func @torch.aten.batch_norm$inference(%arg0: !torch.vtensor<[?,3,?,?],f32>) -> !torch.vtensor<[?,3,?,?],f32> {
  %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %true = torch.constant.bool true
  %false = torch.constant.bool false
  %2 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
  %float1.000000e-01 = torch.constant.float 1.000000e-01
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %int1 = torch.constant.int 1
  %3 = torch.aten.add.Scalar %2, %int1, %int1 : !torch.vtensor<[],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  %4 = torch.aten.batch_norm %arg0, %1, %0, %0, %1, %false, %float1.000000e-01, %float1.000000e-05, %true : !torch.vtensor<[?,3,?,?],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[?,3,?,?],f32>
  return %4 : !torch.vtensor<[?,3,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.relu(
// CHECK-SAME:                          %[[VAL_0:.*]]: !torch.vtensor<[2,5],f32>) -> !torch.vtensor<[2,5],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,5],f32> -> tensor<2x5xf32>
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_2]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x5xf32>
// CHECK:           %[[VAL_4:.*]] = mhlo.maximum %[[VAL_1]], %[[VAL_3]] : tensor<2x5xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<2x5xf32> -> !torch.vtensor<[2,5],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[2,5],f32>
func.func @torch.aten.relu(%arg0: !torch.vtensor<[2,5],f32>) -> !torch.vtensor<[2,5],f32> {
  %0 = torch.aten.relu %arg0 : !torch.vtensor<[2,5],f32> -> !torch.vtensor<[2,5],f32>
  return %0 : !torch.vtensor<[2,5],f32>
}

// CHECK-LABEL:   func.func @torch.aten.relu$dynamic(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = shape.shape_of %[[VAL_1]] : tensor<?x?xf32> -> tensor<2xindex>
// CHECK:           %[[VAL_4:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[VAL_2]], %[[VAL_3]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = mhlo.maximum %[[VAL_1]], %[[VAL_4]] : tensor<?x?xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5:.*]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.relu$dynamic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.relu %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.reciprocal(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[5,5,5],f32> {
// CEHCK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0:.*]] : !torch.vtensor<[5,5,5],f32> -> tensor<5x5x5xf32>
// CEHCK:           %[[VAL_2:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CEHCK:           %[[VAL_3:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_2]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<5x5x5xf32>
// CEHCK:           %[[VAL_4:.*]] = mhlo.divide %[[VAL_3]], %[[VAL_1]] : tensor<5x5x5xf32>
// CEHCK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<5x5x5xf32> -> !torch.vtensor<[5,5,5],f32>
// CEHCK:           return %[[VAL_5]] : !torch.vtensor<[5,5,5],f32>
func.func @torch.aten.reciprocal(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[5,5,5],f32> {
  %0 = torch.aten.reciprocal %arg0 : !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
  return %0 : !torch.vtensor<[5,5,5],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.reciprocal$dynamic(
// CHECK-SAME:                                             %[[VAL_0:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0:.*]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = shape.shape_of %[[VAL_1]] : tensor<?x?x?xf32> -> tensor<3xindex>
// CHECK:           %[[VAL_4:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[VAL_2]], %[[VAL_3]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<3xindex>) -> tensor<?x?x?xf32>
// CHECK:           %[[VAL_5:.*]] = mhlo.divide %[[VAL_4]], %[[VAL_1]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[?,?,?],f32>
func.func @torch.aten.reciprocal$dynamic(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %0 = torch.aten.reciprocal %arg0 : !torch.vtensor<[?,?,?],f32> -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}

// CHECK-LABEL:   func @torch.aten.native_layer_norm(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !torch.vtensor<[3,7,4,5],f32>) -> !torch.vtensor<[3,7,4,5],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,7,4,5],f32> -> tensor<3x7x4x5xf32>
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<4x5xf32>
// CHECK:           %[[VAL_3:.*]] = mhlo.constant dense<1.000000e+00> : tensor<4x5xf32>
// CHECK:           %int4 = torch.constant.int 4
// CHECK:           %int5 = torch.constant.int 5
// CHECK:           %float1.000000e-05 = torch.constant.float 1.000000e-05
// CHECK:           %true = torch.constant.bool true
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %int4, %int5 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = mhlo.constant dense<[1, 21, 20]> : tensor<3xi64>
// CHECK:           %[[VAL_6:.*]] = "mhlo.dynamic_reshape"(%[[VAL_1]], %[[VAL_5]]) : (tensor<3x7x4x5xf32>, tensor<3xi64>) -> tensor<1x21x20xf32>
// CHECK:           %[[VAL_7:.*]] = mhlo.constant dense<1.000000e+00> : tensor<21xf32>
// CHECK:           %[[VAL_8:.*]] = mhlo.constant dense<0.000000e+00> : tensor<21xf32>
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]], %[[VAL_11:.*]] = "mhlo.batch_norm_training"(%[[VAL_6]], %[[VAL_7]], %[[VAL_8]]) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x21x20xf32>, tensor<21xf32>, tensor<21xf32>) -> (tensor<1x21x20xf32>, tensor<21xf32>, tensor<21xf32>)
// CHECK:           %[[VAL_12:.*]] = mhlo.constant dense<[3, 7, 4, 5]> : tensor<4xi64>
// CHECK:           %[[VAL_13:.*]] = "mhlo.dynamic_reshape"(%[[VAL_9]], %[[VAL_12]]) : (tensor<1x21x20xf32>, tensor<4xi64>) -> tensor<3x7x4x5xf32>
// CHECK:           %[[VAL_14:.*]] = mhlo.constant dense<[3, 7, 1, 1]> : tensor<4xi64>
// CHECK:           %[[VAL_15:.*]] = "mhlo.dynamic_reshape"(%[[VAL_10]], %[[VAL_14]]) : (tensor<21xf32>, tensor<4xi64>) -> tensor<3x7x1x1xf32>
// CHECK:           %[[VAL_16:.*]] = mhlo.constant dense<[3, 7, 1, 1]> : tensor<4xi64>
// CHECK:           %[[VAL_17:.*]] = "mhlo.dynamic_reshape"(%[[VAL_11]], %[[VAL_16]]) : (tensor<21xf32>, tensor<4xi64>) -> tensor<3x7x1x1xf32>
// CHECK:           %[[VAL_18:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_3]]) {broadcast_dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<4x5xf32>) -> tensor<3x7x4x5xf32>
// CHECK:           %[[VAL_19:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_2]]) {broadcast_dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<4x5xf32>) -> tensor<3x7x4x5xf32>
// CHECK:           %[[VAL_20:.*]] = mhlo.multiply %[[VAL_13]], %[[VAL_18]] : tensor<3x7x4x5xf32>
// CHECK:           %[[VAL_21:.*]] = mhlo.add %[[VAL_20]], %[[VAL_19]] : tensor<3x7x4x5xf32>
// CHECK:           %[[VAL_22:.*]] = torch_c.from_builtin_tensor %[[VAL_21:.*]] : tensor<3x7x4x5xf32> -> !torch.vtensor<[3,7,4,5],f32>
// CHECK:           return %[[VAL_22]] : !torch.vtensor<[3,7,4,5],f32>
func.func @torch.aten.native_layer_norm(%arg0: !torch.vtensor<[3,7,4,5],f32>) -> !torch.vtensor<[3,7,4,5],f32> {
  %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<4x5xf32>) : !torch.vtensor<[4,5],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<4x5xf32>) : !torch.vtensor<[4,5],f32>
  %int4 = torch.constant.int 4
  %int5 = torch.constant.int 5
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %true = torch.constant.bool true
  %2 = torch.prim.ListConstruct %int4, %int5 : (!torch.int, !torch.int) -> !torch.list<int>
  %result0, %result1, %result2 = torch.aten.native_layer_norm %arg0, %2, %1, %0, %float1.000000e-05 : !torch.vtensor<[3,7,4,5],f32>, !torch.list<int>, !torch.vtensor<[4,5],f32>, !torch.vtensor<[4,5],f32>, !torch.float -> !torch.vtensor<[3,7,4,5],f32>, !torch.vtensor<[3,7,1,1],f32>, !torch.vtensor<[3,7,1,1],f32>
  return %result0 : !torch.vtensor<[3,7,4,5],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.contiguous(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %[[VAL_2:.*]] = torch_c.from_builtin_tensor %[[VAL_1]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:           return %[[VAL_2]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.contiguous(%arg0: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.contiguous %arg0, %int0 : !torch.vtensor<[4,64],f32>, !torch.int -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.prim.NumToTensor.Scalar$basic() -> !torch.vtensor<[],si64> {
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %[[VAL_0:.*]] = mhlo.constant dense<1> : tensor<i64>
// CHECK:           %[[VAL_1:.*]] = torch_c.from_builtin_tensor %[[VAL_0]] : tensor<i64> -> !torch.vtensor<[],si64>
// CHECK:           return %[[VAL_1]] : !torch.vtensor<[],si64>
func.func @torch.prim.NumToTensor.Scalar$basic() -> !torch.vtensor<[], si64> {
  %int1 = torch.constant.int 1
  %0 = torch.prim.NumToTensor.Scalar %int1 : !torch.int -> !torch.vtensor<[], si64>
  return %0 : !torch.vtensor<[], si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.broadcast_to$basic(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[8,4,64],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 64
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 4
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 8
// CHECK:           %[[VAL_5:.*]] = torch.prim.ListConstruct %[[VAL_4]], %[[VAL_3]], %[[VAL_2]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_6:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_1]]) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<4x64xf32>) -> tensor<8x4x64xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<8x4x64xf32> -> !torch.vtensor<[8,4,64],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[8,4,64],f32>
func.func @torch.aten.broadcast_to$basic(%arg0: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[8,4,64],f32> {
  %int64 = torch.constant.int 64
  %int4 = torch.constant.int 4
  %int8 = torch.constant.int 8
  %0 = torch.prim.ListConstruct %int8, %int4, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.broadcast_to %arg0, %0 : !torch.vtensor<[4,64],f32>, !torch.list<int> -> !torch.vtensor<[8,4,64],f32>
  return %1 : !torch.vtensor<[8,4,64],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.broadcast_to$dynamic_implicit(
// CHECK-SAME:                                             %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[8,4,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %int-1 = torch.constant.int -1
// CHECK:           %int4 = torch.constant.int 4
// CHECK:           %int8 = torch.constant.int 8
// CHECK:           %[[VAL_2:.*]] = torch.prim.ListConstruct %int8, %int4, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_i64 %int8
// CHECK:           %[[VAL_4:.*]] = arith.index_cast %[[VAL_3:.*]] : i64 to index
// CHECK:           %[[VAL_5:.*]] = torch_c.to_i64 %int4
// CHECK:           %[[VAL_6:.*]] = arith.index_cast %[[VAL_5]] : i64 to index
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_8:.*]] = tensor.dim %[[VAL_1:.*]], %[[VAL_7]] : tensor<?x?xf32>
// CHECK:           %[[VAL_9:.*]] = tensor.from_elements %[[VAL_4]], %[[VAL_6]], %[[VAL_8]] : tensor<3xindex>
// CHECK:           %[[VAL_10:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[VAL_1]], %[[VAL_9]]) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<3xindex>) -> tensor<8x4x?xf32>
// CHECK:           %[[VAL_11:.*]] = torch_c.from_builtin_tensor %[[VAL_10]] : tensor<8x4x?xf32> -> !torch.vtensor<[8,4,?],f32>
// CHECK:           return %[[VAL_11]] : !torch.vtensor<[8,4,?],f32>
func.func @torch.aten.broadcast_to$dynamic_implicit(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[8,4,?],f32> {
  %int-1 = torch.constant.int -1
  %int4 = torch.constant.int 4
  %int8 = torch.constant.int 8
  %0 = torch.prim.ListConstruct %int8, %int4, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.broadcast_to %arg0, %0 : !torch.vtensor<[?,?],f32>, !torch.list<int> -> !torch.vtensor<[8,4,?],f32>
  return %1 : !torch.vtensor<[8,4,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.permute$basic(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = "mhlo.transpose"(%[[VAL_1]]) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.permute$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?],f32>
  return %1 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.transpose$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = "mhlo.transpose"(%[[VAL_1]]) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.transpose$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.aten.transpose.int %arg0, %int0, %int1 : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}