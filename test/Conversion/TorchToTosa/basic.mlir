// RUN: torch-mlir-opt <%s -convert-torch-to-tosa -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func @torch.aten.tanh$basic(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ARG_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[RESULT_BUILTIN:.*]] = "tosa.tanh"(%[[ARG_BUILTIN]]) : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>
func @torch.aten.tanh$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.tanh %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func @torch.aten.sigmoid$basic(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ARG_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[RESULT_BUILTIN:.*]] = "tosa.sigmoid"(%[[ARG_BUILTIN]]) : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>
func @torch.aten.sigmoid$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.sigmoid %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func @torch.aten.relu$basic(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ARG_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[RESULT_BUILTIN:.*]] = "tosa.clamp"(%[[ARG_BUILTIN]]) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>  
func @torch.aten.relu$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.relu %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}


// -----

// CHECK-LABEL:   func @torch.aten.log$basic(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ARG_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[RESULT_BUILTIN:.*]] = "tosa.log"(%[[ARG_BUILTIN]]) : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>
func @torch.aten.log$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.log %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func @torch.aten.exp$basic(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ARG_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[RESULT_BUILTIN:.*]] = "tosa.exp"(%[[ARG_BUILTIN]]) : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>
func @torch.aten.exp$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.exp %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func @torch.aten.neg$basic(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ARG_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[RESULT_BUILTIN:.*]] = "tosa.negate"(%[[ARG_BUILTIN]]) : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>
func @torch.aten.neg$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.neg %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func @torch.aten.floor$basic(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ARG_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[RESULT_BUILTIN:.*]] = "tosa.floor"(%[[ARG_BUILTIN]]) : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>
func @torch.aten.floor$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.floor %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func @torch.aten.bitwise_not$basic(
// CHECK-SAME:                                %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ARG_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[RESULT_BUILTIN:.*]] = "tosa.bitwise_not"(%[[ARG_BUILTIN]]) : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>
func @torch.aten.bitwise_not$basic(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.bitwise_not %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func @torch.aten.add$basic(
// CHECK-SAME:                               %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                               %[[ARG1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ARG0_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[ARG1_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[ARG2:.*]] = torch.constant.int 1
// CHECK:           %[[RESULT_BUILTIN:.*]] = "tosa.add"(%[[ARG0_BUILTIN]], %[[ARG1_BUILTIN]]) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>  
func @torch.aten.add$basic(%arg0: !torch.vtensor<[?, ?],f32>, %arg1: !torch.vtensor<[?, ?],f32>) -> !torch.vtensor<[?, ?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[?, ?],f32>, !torch.vtensor<[?, ?],f32>, !torch.int -> !torch.vtensor<[?, ?],f32>
  return %0 : !torch.vtensor<[?, ?],f32>
}

// -----

// CHECK-LABEL:   func @torch.aten.sub$basic(
// CHECK-SAME:                               %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                               %[[ARG1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %[[ARG0_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[ARG1_BUILTIN:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[ARG2:.*]] = torch.constant.int 1
// CHECK:           %[[RESULT_BUILTIN:.*]] = "tosa.sub"(%[[ARG0_BUILTIN]], %[[ARG1_BUILTIN]]) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[RESULT_BUILTIN]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[RESULT]] : !torch.vtensor<[?,?],f32>  
func @torch.aten.sub$basic(%arg0: !torch.vtensor<[?, ?],f32>, %arg1: !torch.vtensor<[?, ?],f32>) -> !torch.vtensor<[?, ?],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[?, ?],f32>, !torch.vtensor<[?, ?],f32>, !torch.int -> !torch.vtensor<[?, ?],f32>
  return %0 : !torch.vtensor<[?, ?],f32>
}
