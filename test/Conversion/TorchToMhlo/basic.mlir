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

// CHECK-LABEL:   func.func @torch.aten.transpose$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = "mhlo.transpose"(%[[VAL_1]]) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<4x3xf32>) -> tensor<3x4xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[3,4],f32>
func.func @torch.aten.transpose$basic(%arg0: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[3,4],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.aten.transpose.int %arg0, %int0, %int1 : !torch.vtensor<[4,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.addscalar$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 9
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = mhlo.constant dense<9.000000e+00> : tensor<4x64xf32>
// CHECK:           %[[VAL_5:.*]] = mhlo.add %[[VAL_1]], %[[VAL_4]] : tensor<4x64xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.addscalar$basic(%arg0: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %int9 = torch.constant.int 9
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Scalar %arg0, %int9, %int1 : !torch.vtensor<[4,64],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
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
// CHECK-SAME: 																		  %[[VAL_0:.*]]: !torch.vtensor<[4,64],si32>,
// CHECK-SAME: 																		  %[[VAL_1:.*]]: !torch.vtensor<[4,64],si64>) -> !torch.vtensor<[4,64],si64> {
// CHECK:    			 %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],si32> -> tensor<4x64xi32>
// CHECK:    			 %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,64],si64> -> tensor<4x64xi64>
// CHECK:    			 %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:    			 %[[VAL_5:.*]] = mhlo.convert(%[[VAL_2]]) : (tensor<4x64xi32>) -> tensor<4x64xi64>
// CHECK:    			 %[[VAL_6:.*]] = mhlo.add %[[VAL_5]], %[[VAL_3]] : tensor<4x64xi64>
// CHECK:    			 %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<4x64xi64> -> !torch.vtensor<[4,64],si64>
// CHECK:    			 return %[[VAL_7]] : !torch.vtensor<[4,64],si64>
func.func @torch.aten.addtensor$promote(%arg0: !torch.vtensor<[4,64],si32>, %arg1: !torch.vtensor<[4,64],si64>) -> !torch.vtensor<[4,64],si64> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[4,64],si32>, !torch.vtensor<[4,64],si64>, !torch.int -> !torch.vtensor<[4,64],si64>
  return %0 : !torch.vtensor<[4,64],si64>
}

// -----

// CHECK-LABEL:  	func.func @torch.aten.addtensor$bcast(
// CHECK-SAME: 														         %[[VAL_0:.*]]: !torch.vtensor<[64],f32>,
// CHECK-SAME: 																	   %[[VAL_1:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:    				%[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[64],f32> -> tensor<64xf32>
// CHECK:    				%[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:    				%[[VAL_4:.*]] = torch.constant.int 1
// CHECK:    				%[[VAL_5:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_2]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<4x64xf32>
// CHECK:    				%[[VAL_6:.*]] = mhlo.add %[[VAL_5]], %[[VAL_3]] : tensor<4x64xf32>
// CHECK:    				%[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:    				return %[[VAL_7]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.addtensor$bcast(%arg0: !torch.vtensor<[64],f32>, %arg1: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[64],f32>, !torch.vtensor<[4,64],f32>, !torch.int -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
}

// -----

// CHECK-LABEL: 	func.func @torch.aten.addtensor$alpha(
// CHECK-SAME: 																 		 %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>,
// CHECK-SAME: 																 		 %[[VAL_1:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:    			  %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:    			  %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:    			  %[[VAL_4:.*]] = torch.constant.int 2
// CHECK:    			  %[[VAL_5:.*]] = mhlo.constant dense<2.000000e+00> : tensor<4x64xf32>
// CHECK:    			  %[[VAL_6:.*]] = mhlo.multiply %[[VAL_3]], %[[VAL_5]] : tensor<4x64xf32>
// CHECK:    			  %[[VAL_7:.*]] = mhlo.add %[[VAL_2]], %[[VAL_6]] : tensor<4x64xf32>
// CHECK:    			  %[[VAL_8:.*]] = torch_c.from_builtin_tensor %4 : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:    			  return %[[VAL_8]] : !torch.vtensor<[4,64],f32>
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
// CHECK:           %[[VAL_3:.*]] = mhlo.constant dense<9.000000e+00> : tensor<4x64xf32>
// CHECK:           %[[VAL_4:.*]] = mhlo.multiply %[[VAL_1]], %[[VAL_3]] : tensor<4x64xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[4,64],f32>
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
// CHECK-SAME:																		%[[VAL_0:.*]]: !torch.vtensor<[8,4,64],f32>,
// CHECK-SAME:																		%[[VAL_1:.*]]: !torch.vtensor<[8,1,64],f32>) -> !torch.vtensor<[8,4,64],f32> {
// CHECK:    			 %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[8,4,64],f32> -> tensor<8x4x64xf32>
// CHECK:    			 %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[8,1,64],f32> -> tensor<8x1x64xf32>
// CHECK:    			 %[[VAL_4:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_3]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<8x1x64xf32>) -> tensor<8x4x64xf32>
// CHECK:    			 %[[VAL_5:.*]] = mhlo.multiply %[[VAL_2]], %[[VAL_4]] : tensor<8x4x64xf32>
// CHECK:    			 %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<8x4x64xf32> -> !torch.vtensor<[8,4,64],f32>
// CHECK:    			 return %[[VAL_6]] : !torch.vtensor<[8,4,64],f32>
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
// CHECK:           %[[VAL_4:.*]] = mhlo.constant dense<9.000000e+00> : tensor<4x64xf32>
// CHECK:           %[[VAL_5:.*]] = mhlo.subtract %[[VAL_1]], %[[VAL_4]] : tensor<4x64xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[4,64],f32>
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
// CHECK-SAME: 																		  %[[VAL_0:.*]]: !torch.vtensor<[4,64],si32>,
// CHECK-SAME: 																		  %[[VAL_1:.*]]: !torch.vtensor<[4,64],si64>) -> !torch.vtensor<[4,64],si64> {
// CHECK:    			 %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],si32> -> tensor<4x64xi32>
// CHECK:    			 %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,64],si64> -> tensor<4x64xi64>
// CHECK:    			 %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:    			 %[[VAL_5:.*]] = mhlo.convert(%[[VAL_2]]) : (tensor<4x64xi32>) -> tensor<4x64xi64>
// CHECK:    			 %[[VAL_6:.*]] = mhlo.subtract %[[VAL_5]], %[[VAL_3]] : tensor<4x64xi64>
// CHECK:    			 %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<4x64xi64> -> !torch.vtensor<[4,64],si64>
// CHECK:    			 return %[[VAL_7]] : !torch.vtensor<[4,64],si64>
func.func @torch.aten.subtensor$promote(%arg0: !torch.vtensor<[4,64],si32>, %arg1: !torch.vtensor<[4,64],si64>) -> !torch.vtensor<[4,64],si64> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[4,64],si32>, !torch.vtensor<[4,64],si64>, !torch.int -> !torch.vtensor<[4,64],si64>
  return %0 : !torch.vtensor<[4,64],si64>
}

// -----

// CHECK-LABEL:  	func.func @torch.aten.subtensor$bcast(
// CHECK-SAME: 														         %[[VAL_0:.*]]: !torch.vtensor<[64],f32>,
// CHECK-SAME: 																	   %[[VAL_1:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:    				%[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[64],f32> -> tensor<64xf32>
// CHECK:    				%[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:    				%[[VAL_4:.*]] = torch.constant.int 1
// CHECK:    				%[[VAL_5:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_2]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<4x64xf32>
// CHECK:    				%[[VAL_6:.*]] = mhlo.subtract %[[VAL_5]], %[[VAL_3]] : tensor<4x64xf32>
// CHECK:    				%[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:    				return %[[VAL_7]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.subtensor$bcast(%arg0: !torch.vtensor<[64],f32>, %arg1: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[64],f32>, !torch.vtensor<[4,64],f32>, !torch.int -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
}

// -----

// CHECK-LABEL: 	func.func @torch.aten.subtensor$alpha(
// CHECK-SAME: 																 		 %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>,
// CHECK-SAME: 																 		 %[[VAL_1:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:    			  %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:    			  %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:    			  %[[VAL_4:.*]] = torch.constant.int 2
// CHECK:    			  %[[VAL_5:.*]] = mhlo.constant dense<2.000000e+00> : tensor<4x64xf32>
// CHECK:    			  %[[VAL_6:.*]] = mhlo.multiply %[[VAL_3]], %[[VAL_5]] : tensor<4x64xf32>
// CHECK:    			  %[[VAL_7:.*]] = mhlo.subtract %[[VAL_2]], %[[VAL_6]] : tensor<4x64xf32>
// CHECK:    			  %[[VAL_8:.*]] = torch_c.from_builtin_tensor %4 : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:    			  return %[[VAL_8]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.subtensor$alpha(%arg0: !torch.vtensor<[4,64],f32>, %arg1: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %int2 = torch.constant.int 2
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %int2 : !torch.vtensor<[4,64],f32>, !torch.vtensor<[4,64],f32>, !torch.int -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.divscalar$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 9
// CHECK:           %[[VAL_3:.*]] = mhlo.constant dense<9.000000e+00> : tensor<4x64xf32>
// CHECK:           %[[VAL_4:.*]] = mhlo.divide %[[VAL_1]], %[[VAL_3]] : tensor<4x64xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.divscalar$basic(%arg0: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %int9 = torch.constant.int 9
  %0 = torch.aten.div.Scalar %arg0, %int9 : !torch.vtensor<[4,64],f32>, !torch.int -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.divtensor$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
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

// CHECK-LABEL:  func.func @torch.aten.divtensor$bcast(
// CHECK-SAME:																		%[[VAL_0:.*]]: !torch.vtensor<[8,4,64],f32>,
// CHECK-SAME:																		%[[VAL_1:.*]]: !torch.vtensor<[8,1,64],f32>) -> !torch.vtensor<[8,4,64],f32> {
// CHECK:    			 %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[8,4,64],f32> -> tensor<8x4x64xf32>
// CHECK:    			 %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[8,1,64],f32> -> tensor<8x1x64xf32>
// CHECK:    			 %[[VAL_4:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_3]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<8x1x64xf32>) -> tensor<8x4x64xf32>
// CHECK:    			 %[[VAL_5:.*]] = mhlo.divide %[[VAL_2]], %[[VAL_4]] : tensor<8x4x64xf32>
// CHECK:    			 %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<8x4x64xf32> -> !torch.vtensor<[8,4,64],f32>
// CHECK:    			 return %[[VAL_6]] : !torch.vtensor<[8,4,64],f32>
func.func @torch.aten.divtensor$bcast(%arg0: !torch.vtensor<[8,4,64],f32>, %arg1: !torch.vtensor<[8,1,64],f32>) -> !torch.vtensor<[8,4,64],f32> {
  %0 = torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[8,4,64],f32>, !torch.vtensor<[8,1,64],f32> -> !torch.vtensor<[8,4,64],f32>
  return %0 : !torch.vtensor<[8,4,64],f32>
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

// CHECK-LABEL:   func.func @torch.aten.permute$basic(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[64,4],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = "mhlo.transpose"(%[[VAL_1]]) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<4x64xf32>) -> tensor<64x4xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<64x4xf32> -> !torch.vtensor<[64,4],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[64,4],f32>
func.func @torch.aten.permute$basic(%arg0: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[64,4],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[4,64],f32>, !torch.list<int> -> !torch.vtensor<[64,4],f32>
  return %1 : !torch.vtensor<[64,4],f32>
}


// -----

// CHECK-LABEL:   func.func @torch.aten.clone$basic(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.none
// CHECK:           %[[VAL_3:.*]] = "mhlo.copy"(%[[VAL_1]]) : (tensor<4x64xf32>) -> tensor<4x64xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<4x64xf32> -> !torch.vtensor<[4,64],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[4,64],f32>
func.func @torch.aten.clone$basic(%arg0: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],f32> {
  %none = torch.constant.none
  %0 = torch.aten.clone %arg0, %none : !torch.vtensor<[4,64],f32>, !torch.none -> !torch.vtensor<[4,64],f32>
  return %0 : !torch.vtensor<[4,64],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.slice.Tensor$basic(
// CHECK-SAME:                                       %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[2,64],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_5:.*]] = "mhlo.slice"(%[[VAL_1]]) {limit_indices = dense<[2, 64]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x64xf32>) -> tensor<2x64xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<2x64xf32> -> !torch.vtensor<[2,64],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[2,64],f32>
func.func @torch.aten.slice.Tensor$basic(%arg0: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[2,64],f32> {
  %int2 = torch.constant.int 2
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.aten.slice.Tensor %arg0, %int0, %int0, %int2, %int1 : !torch.vtensor<[4,64],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[2,64],f32>
  return %0 : !torch.vtensor<[2,64],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.atan.view$basic(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[16,16],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 16
// CHECK:           %[[VAL_3:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_4:.*]] = "mhlo.reshape"(%[[VAL_1]]) : (tensor<4x64xf32>) -> tensor<16x16xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<16x16xf32> -> !torch.vtensor<[16,16],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[16,16],f32>
func.func @torch.atan.view$basic(%arg0: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[16,16],f32> {
  %int16 = torch.constant.int 16
  %0 = torch.prim.ListConstruct %int16, %int16 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.view %arg0, %0 : !torch.vtensor<[4,64],f32>, !torch.list<int> -> !torch.vtensor<[16,16],f32>
  return %1 : !torch.vtensor<[16,16],f32>
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

// CHECK-LABEL:   func.func @torch.aten.index_select$basic(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[2,64],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<[1, 0]> : tensor<2xi64>
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_4:.*]] = "mhlo.torch_index_select"(%[[VAL_1]], %[[VAL_2]]) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<4x64xf32>, tensor<2xi64>) -> tensor<2x64xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<2x64xf32> -> !torch.vtensor<[2,64],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[2,64],f32>
func.func @torch.aten.index_select$basic(%arg0: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[2,64],f32> {
  %0 = torch.vtensor.literal(dense<[1, 0]> : tensor<2xsi64>) : !torch.vtensor<[2],si64>
  %int0 = torch.constant.int 0
  %1 = torch.aten.index_select %arg0, %int0, %0 : !torch.vtensor<[4,64],f32>, !torch.int, !torch.vtensor<[2],si64> -> !torch.vtensor<[2,64],f32>
  return %1 : !torch.vtensor<[2,64],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.gt.scalar(
// CHECK-SAME:                             %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],i1> {
// CHECK:         %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:         %[[VAL_2:.*]] = torch.constant.int 3
// CHECK:         %[[VAL_3:.*]] = mhlo.constant dense<3.000000e+00> : tensor<f32>
// CHECK:         %[[VAL_4:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_3]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<4x64xf32>
// CHECK:         %[[VAL_5:.*]] = "mhlo.compare"(%[[VAL_1]], %[[VAL_4]]) {compare_type = #mhlo<"comparison_type FLOAT">, comparison_direction = #mhlo<"comparison_direction GT">} : (tensor<4x64xf32>, tensor<4x64xf32>) -> tensor<4x64xi1>
// CHECK:         %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<4x64xi1> -> !torch.vtensor<[4,64],i1>
// CHECK:         return %[[VAL_6]] : !torch.vtensor<[4,64],i1>
func.func @torch.aten.gt.scalar(%arg0: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[4,64],i1> {
  %int3 = torch.constant.int 3
  %0 = torch.aten.gt.Scalar %arg0, %int3 : !torch.vtensor<[4,64],f32>, !torch.int -> !torch.vtensor<[4,64],i1>
  return %0 : !torch.vtensor<[4,64],i1>
}

// -----

// CHECK-LABEL:    func.func @torch.aten.gt.tensor(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>, 
// CHECK-SAME:                                %[[VAL_1:.*]]: !torch.vtensor<[64],f32>) -> !torch.vtensor<[4,64],i1> {
// CHECK:            %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:            %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[64],f32> -> tensor<64xf32>
// CHECK:            %[[VAL_4:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_3]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<4x64xf32>
// CHECK:            %[[VAL_5:.*]] = "mhlo.compare"(%[[VAL_2]], %[[VAL_4]]) {compare_type = #mhlo<"comparison_type FLOAT">, comparison_direction = #mhlo<"comparison_direction GT">} : (tensor<4x64xf32>, tensor<4x64xf32>) -> tensor<4x64xi1>
// CHECK:            %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<4x64xi1> -> !torch.vtensor<[4,64],i1>
// CHECK:            return %[[VAL_6]] : !torch.vtensor<[4,64],i1>
func.func @torch.aten.gt.tensor(%arg0: !torch.vtensor<[4,64],f32>, %arg1: !torch.vtensor<[64],f32>) -> !torch.vtensor<[4,64],i1> {
  %0 = torch.aten.gt.Tensor %arg0, %arg1 : !torch.vtensor<[4,64],f32>, !torch.vtensor<[64],f32> -> !torch.vtensor<[4,64],i1>
  return %0 : !torch.vtensor<[4,64],i1>
}

// -----

// CHECK-LABEL:    func.func @torch.aten.gt.tensor$convert(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[4,64],si32>, 
// CHECK-SAME:                                        %[[VAL_1:.*]]: !torch.vtensor<[64],f32>) -> !torch.vtensor<[4,64],i1> {
// CHECK:            %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],si32> -> tensor<4x64xi32>
// CHECK:            %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[64],f32> -> tensor<64xf32>
// CHECK:            %[[VAL_4:.*]] = mhlo.convert(%[[VAL_3]]) : (tensor<64xf32>) -> tensor<64xi32>
// CHECK:            %[[VAL_5:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_4]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<64xi32>) -> tensor<4x64xi32>
// CHECK:            %[[VAL_6:.*]] = "mhlo.compare"(%[[VAL_2]], %[[VAL_5]]) {compare_type = #mhlo<"comparison_type SIGNED">, comparison_direction = #mhlo<"comparison_direction GT">} : (tensor<4x64xi32>, tensor<4x64xi32>) -> tensor<4x64xi1>
// CHECK:            %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<4x64xi1> -> !torch.vtensor<[4,64],i1>
// CHECK:            return %[[VAL_7]] : !torch.vtensor<[4,64],i1>

func.func @torch.aten.gt.tensor$convert(%arg0: !torch.vtensor<[4,64],si32>, %arg1: !torch.vtensor<[64],f32>) -> !torch.vtensor<[4,64],i1> {
  %0 = torch.aten.gt.Tensor %arg0, %arg1 : !torch.vtensor<[4,64],si32>, !torch.vtensor<[64],f32> -> !torch.vtensor<[4,64],i1>
  return %0 : !torch.vtensor<[4,64],i1>
}

// -----

// CHECK-LABEL:    func.func @torch.aten.lt.tensor(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>, 
// CHECK-SAME:                                %[[VAL_1:.*]]: !torch.vtensor<[64],f32>) -> !torch.vtensor<[4,64],i1> {
// CHECK:            %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:            %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[64],f32> -> tensor<64xf32>
// CHECK:            %[[VAL_4:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_3]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<4x64xf32>
// CHECK:            %[[VAL_5:.*]] = "mhlo.compare"(%[[VAL_2]], %[[VAL_4]]) {compare_type = #mhlo<"comparison_type FLOAT">, comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<4x64xf32>, tensor<4x64xf32>) -> tensor<4x64xi1>
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
// CHECK:            %[[VAL_5:.*]] = "mhlo.compare"(%[[VAL_2]], %[[VAL_4]]) {compare_type = #mhlo<"comparison_type FLOAT">, comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<4x64xf32>, tensor<4x64xf32>) -> tensor<4x64xi1>
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
// CHECK:            %[[VAL_5:.*]] = "mhlo.compare"(%[[VAL_2]], %[[VAL_4]]) {compare_type = #mhlo<"comparison_type FLOAT">, comparison_direction = #mhlo<"comparison_direction NE">} : (tensor<4x64xf32>, tensor<4x64xf32>) -> tensor<4x64xi1>
// CHECK:            %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<4x64xi1> -> !torch.vtensor<[4,64],i1>
// CHECK:            return %[[VAL_6]] : !torch.vtensor<[4,64],i1>
func.func @torch.aten.ne.tensor(%arg0: !torch.vtensor<[4,64],f32>, %arg1: !torch.vtensor<[64],f32>) -> !torch.vtensor<[4,64],i1> {
  %0 = torch.aten.ne.Tensor %arg0, %arg1 : !torch.vtensor<[4,64],f32>, !torch.vtensor<[64],f32> -> !torch.vtensor<[4,64],i1>
  return %0 : !torch.vtensor<[4,64],i1>
}

// ----- 
// CHECK-LABEL:   func.func @torch.aten.unsqueeze(
// CHECK-SAME:                  %[[VAL_0:.*]]: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[1,4,64],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,64],f32> -> tensor<4x64xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_3:.*]] = mhlo.constant dense<[1, 4, 64]> : tensor<3xi64>
// CHECK:           %[[VAL_4:.*]] = "mhlo.dynamic_reshape"(%[[VAL_1]], %[[VAL_3]]) : (tensor<4x64xf32>, tensor<3xi64>) -> tensor<1x4x64xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<1x4x64xf32> -> !torch.vtensor<[1,4,64],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[1,4,64],f32>
func.func @torch.aten.unsqueeze(%arg0: !torch.vtensor<[4,64],f32>) -> !torch.vtensor<[1,4,64],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.unsqueeze %arg0, %int0 : !torch.vtensor<[4,64],f32>, !torch.int -> !torch.vtensor<[1,4,64],f32>
  return %0 : !torch.vtensor<[1,4,64],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.unsqueeze$trailing(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3,1],f32> {
// CEHCK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,3],f32> -> tensor<2x3xf32>
// CEHCK:           %int2 = torch.constant.int 2
// CEHCK:           %[[VAL_2:.*]] = mhlo.constant dense<[2, 3, 1]> : tensor<3xi64>
// CEHCK:           %[[VAL_3:.*]] = "mhlo.dynamic_reshape"(%[[VAL_1]], %[[VAL_2]]) : (tensor<2x3xf32>, tensor<3xi64>) -> tensor<2x3x1xf32>
// CEHCK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<2x3x1xf32> -> !torch.vtensor<[2,3,1],f32>
// CEHCK:           return %[[VAL_4]] : !torch.vtensor<[2,3,1],f32>
func.func @torch.aten.unsqueeze$trailing(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3,1],f32> {
  %int2 = torch.constant.int 2
  %0 = torch.aten.unsqueeze %arg0, %int2 : !torch.vtensor<[2,3],f32>, !torch.int -> !torch.vtensor<[2,3,1],f32>
  return %0 : !torch.vtensor<[2,3,1],f32>
}

//-----

// CHECK-LABEL:   func.func @torch.aten.unsqueeze$negative_dim(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,3,4,1],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,3,4],f32> -> tensor<2x3x4xf32>
// CHECK:           %int-1 = torch.constant.int -1
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<[2, 3, 4, 1]> : tensor<4xi64>
// CHECK:           %[[VAL_3:.*]] = "mhlo.dynamic_reshape"(%[[VAL_1]], %[[VAL_2]]) : (tensor<2x3x4xf32>, tensor<4xi64>) -> tensor<2x3x4x1xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<2x3x4x1xf32> -> !torch.vtensor<[2,3,4,1],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[2,3,4,1],f32>
func.func @torch.aten.unsqueeze$negative_dim(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,3,4,1],f32> {
  %int-1 = torch.constant.int -1
  %0 = torch.aten.unsqueeze %arg0, %int-1 : !torch.vtensor<[2,3,4],f32>, !torch.int -> !torch.vtensor<[2,3,4,1],f32>
  return %0 : !torch.vtensor<[2,3,4,1],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.batch_norm(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[2,3,5,5],f32>) -> !torch.vtensor<[2,3,5,5],f32> {
// CEHCK：          %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,3,5,5],f32> -> tensor<2x3x5x5xf32>
// CEHCK：          %[[VAL_2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<3xf32>
// CEHCK：          %[[VAL_3:.*]] = mhlo.constant dense<1.000000e+00> : tensor<3xf32>
// CEHCK：          %true = torch.constant.bool true
// CEHCK：          %[[VAL_4:.*]] = mhlo.constant dense<0> : tensor<i64>
// CEHCK：          %float1.000000e-01 = torch.constant.float 1.000000e-01
// CEHCK：          %float1.000000e-05 = torch.constant.float 1.000000e-05
// CEHCK：          %int1 = torch.constant.int 1
// CEHCK：          %[[VAL_5:.*]] = mhlo.constant dense<1> : tensor<i64>
// CEHCK：          %[[VAL_6:.*]] = mhlo.add %[[VAL_4]], %[[VAL_5]] : tensor<i64>
// CEHCK：          %[[VAL_7:.*]], %batch_mean, %batch_var = "mhlo.batch_norm_training"(%[[VAL_1]], %[[VAL_3]], %[[VAL_2]]) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<2x3x5x5xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<2x3x5x5xf32>, tensor<3xf32>, tensor<3xf32>)
// CEHCK：          %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<2x3x5x5xf32> -> !torch.vtensor<[2,3,5,5],f32>
// CEHCK：          return %[[VAL_8]] : !torch.vtensor<[2,3,5,5],f32>

func.func @torch.aten.batch_norm(%arg0: !torch.vtensor<[2,3,5,5],f32>) -> !torch.vtensor<[2,3,5,5],f32> {
  %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %true = torch.constant.bool true
  %2 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
  %float1.000000e-01 = torch.constant.float 1.000000e-01
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %int1 = torch.constant.int 1
  %3 = torch.aten.add.Scalar %2, %int1, %int1 : !torch.vtensor<[],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  %4 = torch.aten.batch_norm %arg0, %1, %0, %0, %1, %true, %float1.000000e-01, %float1.000000e-05, %true : !torch.vtensor<[2,3,5,5],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[2,3,5,5],f32>
  return %4 : !torch.vtensor<[2,3,5,5],f32>
}

// CHECK-LABEL:   func.func @torch.aten.batch_norm$none_bias_weight(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: !torch.vtensor<[2,3,5,5],f32>) -> !torch.vtensor<[2,3,5,5],f32> {
// CEHCK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,3,5,5],f32> -> tensor<2x3x5x5xf32>
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
// CEHCK:           %[[VAL_7:.*]], %batch_mean, %batch_var = "mhlo.batch_norm_training"(%[[VAL_1]], %[[VAL_5]], %[[VAL_6]]) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<2x3x5x5xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<2x3x5x5xf32>, tensor<3xf32>, tensor<3xf32>)
// CEHCK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<2x3x5x5xf32> -> !torch.vtensor<[2,3,5,5],f32>
// CEHCK:           return %[[VAL_8]] : !torch.vtensor<[2,3,5,5],f32>
func.func @torch.aten.batch_norm$none_bias_weight(%arg0: !torch.vtensor<[2,3,5,5],f32>) -> !torch.vtensor<[2,3,5,5],f32> {
  %none = torch.constant.none
  %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %1 = torch.vtensor.literal(dense<0.000000e+00> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %true = torch.constant.bool true
  %2 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
  %float1.000000e-01 = torch.constant.float 1.000000e-01
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %int1 = torch.constant.int 1
  %3 = torch.aten.add.Scalar %2, %int1, %int1 : !torch.vtensor<[],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  %4 = torch.aten.batch_norm %arg0, %none, %none, %1, %0, %true, %float1.000000e-01, %float1.000000e-05, %true : !torch.vtensor<[2,3,5,5],f32>, !torch.none, !torch.none, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[2,3,5,5],f32>
  return %4 : !torch.vtensor<[2,3,5,5],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.batch_norm$inference(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[2,3,5,5],f32>) -> !torch.vtensor<[2,3,5,5],f32> {
// CEHCK：          %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,3,5,5],f32> -> tensor<2x3x5x5xf32>
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
// CEHCK：          %[[VAL_7:.*]], %batch_mean, %batch_var = "mhlo.batch_norm_training"(%[[VAL_1]], %[[VAL_3]], %[[VAL_2]], %[[VAL_2]],  %[[VAL_3]]) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<2x3x5x5xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<2x3x5x5xf32>, tensor<3xf32>, tensor<3xf32>)
// CEHCK：          %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<2x3x5x5xf32> -> !torch.vtensor<[2,3,5,5],f32>
// CEHCK：          return %[[VAL_8]] : !torch.vtensor<[2,3,5,5],f32>
func.func @torch.aten.batch_norm$inference(%arg0: !torch.vtensor<[2,3,5,5],f32>) -> !torch.vtensor<[2,3,5,5],f32> {
  %0 = torch.vtensor.literal(dense<0.000000e+00> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %true = torch.constant.bool true
  %false = torch.constant.bool false
  %2 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
  %float1.000000e-01 = torch.constant.float 1.000000e-01
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %int1 = torch.constant.int 1
  %3 = torch.aten.add.Scalar %2, %int1, %int1 : !torch.vtensor<[],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  %4 = torch.aten.batch_norm %arg0, %1, %0, %0, %1, %false, %float1.000000e-01, %float1.000000e-05, %true : !torch.vtensor<[2,3,5,5],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.bool, !torch.float, !torch.float, !torch.bool -> !torch.vtensor<[2,3,5,5],f32>
  return %4 : !torch.vtensor<[2,3,5,5],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.squeeze.dim(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !torch.vtensor<[2,1,5,1],f32>) -> !torch.vtensor<[2,5],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,1,5,1],f32> -> tensor<2x1x5x1xf32>
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<[2, 5]> : tensor<2xi64>
// CHECK:           %[[VAL_3:.*]] = "mhlo.dynamic_reshape"(%[[VAL_1]], %[[VAL_2]]) : (tensor<2x1x5x1xf32>, tensor<2xi64>) -> tensor<2x5xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<2x5xf32> -> !torch.vtensor<[2,5],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[2,5],f32>
func.func @torch.aten.squeeze.dim(%arg0: !torch.vtensor<[2,1,5,1],f32>) -> !torch.vtensor<[2,5],f32> {
  %0 = torch.aten.squeeze %arg0 : !torch.vtensor<[2,1,5,1],f32> -> !torch.vtensor<[2,5],f32>
  return %0 : !torch.vtensor<[2,5],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.squeeze(
// CHECK-SAME:                             %[[VAL_0:.*]]: !torch.vtensor<[2,1,5,1],f32>) -> !torch.vtensor<[2,5],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,1,5,1],f32> -> tensor<2x1x5x1xf32>
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<[2, 5]> : tensor<2xi64>
// CHECK:           %[[VAL_3:.*]] = "mhlo.dynamic_reshape"(%[[VAL_1]], %[[VAL_2]]) : (tensor<2x1x5x1xf32>, tensor<2xi64>) -> tensor<2x5xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<2x5xf32> -> !torch.vtensor<[2,5],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[2,5],f32>
func.func @torch.aten.squeeze(%arg0: !torch.vtensor<[2,1,5,1],f32>) -> !torch.vtensor<[2,5],f32> {
  %0 = torch.aten.squeeze %arg0 : !torch.vtensor<[2,1,5,1],f32> -> !torch.vtensor<[2,5],f32>
  return %0 : !torch.vtensor<[2,5],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.relu(
// CHECK-SAME:                          %[[VAL_0:.*]]: !torch.vtensor<[2,5],f32>) -> !torch.vtensor<[2,5],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,5],f32> -> tensor<2x5xf32>
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<2x5xf32>
// CHECK:           %[[VAL_3:.*]] = mhlo.maximum %[[VAL_1]], %[[VAL_2]] : tensor<2x5xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<2x5xf32> -> !torch.vtensor<[2,5],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[2,5],f32>
func.func @torch.aten.relu(%arg0: !torch.vtensor<[2,5],f32>) -> !torch.vtensor<[2,5],f32> {
  %0 = torch.aten.relu %arg0 : !torch.vtensor<[2,5],f32> -> !torch.vtensor<[2,5],f32>
  return %0 : !torch.vtensor<[2,5],f32>
}
// -----
// CHECK-LABEL:   func.func @torch.aten.relu$int8(
// CHECK-SAME:                               %[[VAL_0:.*]]: !torch.vtensor<[2,5],si8>) -> !torch.vtensor<[2,5],si8> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,5],si8> -> tensor<2x5xi8>
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<0> : tensor<2x5xi8>
// CHECK:           %[[VAL_3:.*]] = mhlo.maximum %[[VAL_1]], %[[VAL_2]] : tensor<2x5xi8>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<2x5xi8> -> !torch.vtensor<[2,5],si8>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[2,5],si8>
func.func @torch.aten.relu$int8(%arg0: !torch.vtensor<[2,5],si8>) -> !torch.vtensor<[2,5],si8> {
  %0 = torch.aten.relu %arg0 : !torch.vtensor<[2,5],si8> -> !torch.vtensor<[2,5],si8>
  return %0 : !torch.vtensor<[2,5],si8>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.flatten.using_ints(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[24],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,3,4],f32> -> tensor<2x3x4xf32>
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %int-1 = torch.constant.int -1
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<24> : tensor<1xi64>
// CHECK:           %[[VAL_3:.*]] = "mhlo.dynamic_reshape"(%[[VAL_1]], %[[VAL_2]]) : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<24xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<24xf32> -> !torch.vtensor<[24],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[24],f32>
func.func @torch.aten.flatten.using_ints(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[24],f32> {
  %int0 = torch.constant.int 0
  %int-1 = torch.constant.int -1
  %0 = torch.aten.flatten.using_ints %arg0, %int0, %int-1 : !torch.vtensor<[2,3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[24],f32>
  return %0 : !torch.vtensor<[24],f32>
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