// RUN: torch-mlir-opt <%s -convert-torch-to-mhlo -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.max.dim$keepdim(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>) -> (!torch.vtensor<[?,1],f32>, !torch.vtensor<[?,1],si64>) {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:         %[[TRUE:.*]] = torch.constant.bool true
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM_0:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[DIM_0]] : index to i64
// CHECK:         %[[T3:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:         %[[T4:.*]] = mhlo.constant dense<0> : tensor<i64>
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[T1]], %[[T2]] : tensor<2xi64>
// CHECK:         %[[T5:.*]] = "mhlo.dynamic_iota"(%[[FROM_ELEMENTS]]) {iota_dimension = 1 : i64} : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:         %[[T6:.*]]:2 = mhlo.reduce(%[[T0]] init: %[[T3]]), (%[[T5]] init: %[[T4]]) across dimensions = [1] : (tensor<?x?xf32>, tensor<?x?xi64>, tensor<f32>, tensor<i64>) -> (tensor<?xf32>, tensor<?xi64>)
// CHECK:         reducer(%[[ARG1:.*]]: tensor<f32>, %[[ARG3:.*]]: tensor<f32>) (%[[ARG2:.*]]: tensor<i64>, %[[ARG4:.*]]: tensor<i64>)  {
// CHECK:         %[[T11:.*]] = mhlo.compare  GE, %[[ARG1]], %[[ARG3]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:         %[[T12:.*]] = mhlo.select %[[T11]], %[[ARG1]], %[[ARG3]] : tensor<i1>, tensor<f32>
// CHECK:         %[[T13:.*]] = mhlo.compare  EQ, %[[ARG1]], %[[ARG3]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:         %[[T14:.*]] = mhlo.minimum %[[ARG2]], %[[ARG4]] : tensor<i64>
// CHECK:         %[[T15:.*]] = mhlo.select %[[T11]], %[[ARG2]], %[[ARG4]] : tensor<i1>, tensor<i64>
// CHECK:         %[[T16:.*]] = mhlo.select %[[T13]], %[[T14]], %[[T15]] : tensor<i1>, tensor<i64>
// CHECK:         mhlo.return %[[T12]], %[[T16]] : tensor<f32>, tensor<i64>
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[FROM_ELEMENTS_1:.*]] = tensor.from_elements %[[T1]], %[[C1_I64]] : tensor<2xi64>
// CHECK:         %[[T7:.*]] = mhlo.dynamic_reshape %[[T6]]#0, %[[FROM_ELEMENTS_1]] : (tensor<?xf32>, tensor<2xi64>) -> tensor<?x1xf32>
// CHECK:         %[[T8:.*]] = mhlo.dynamic_reshape %[[T6]]#1, %[[FROM_ELEMENTS_1]] : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
// CHECK:         %[[T9:.*]] = torch_c.from_builtin_tensor %[[T7]] : tensor<?x1xf32> -> !torch.vtensor<[?,1],f32>
// CHECK:         %[[T10:.*]] = torch_c.from_builtin_tensor %[[T8]] : tensor<?x1xi64> -> !torch.vtensor<[?,1],si64>
// CHECK:         return %[[T9]], %[[T10]] : !torch.vtensor<[?,1],f32>, !torch.vtensor<[?,1],si64>
func.func @torch.aten.max.dim$keepdim(%arg0: !torch.vtensor<[?,?],f32>) -> (!torch.vtensor<[?,1],f32>, !torch.vtensor<[?,1],si64>) {
  %true = torch.constant.bool true
  %int1 = torch.constant.int 1
  %values, %indices = torch.aten.max.dim %arg0, %int1, %true : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?,1],f32>, !torch.vtensor<[?,1],si64>
  return %values, %indices : !torch.vtensor<[?,1],f32>, !torch.vtensor<[?,1],si64>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.max.dim(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>) -> (!torch.vtensor<[?],f32>, !torch.vtensor<[?],si64>) {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:         %[[FALSE:.*]] = torch.constant.bool false
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM_0:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[DIM_0]] : index to i64
// CHECK:         %[[T3:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:         %[[T4:.*]] = mhlo.constant dense<0> : tensor<i64>
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[T1]], %[[T2]] : tensor<2xi64>
// CHECK:         %[[T5:.*]] = "mhlo.dynamic_iota"(%[[FROM_ELEMENTS]]) {iota_dimension = 1 : i64} : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:         %[[T6:.*]]:2 = mhlo.reduce(%[[T0]] init: %[[T3]]), (%[[T5]] init: %[[T4]]) across dimensions = [1] : (tensor<?x?xf32>, tensor<?x?xi64>, tensor<f32>, tensor<i64>) -> (tensor<?xf32>, tensor<?xi64>)
// CHECK:         reducer(%[[ARG1:.*]]: tensor<f32>, %[[ARG3:.*]]: tensor<f32>) (%[[ARG2:.*]]: tensor<i64>, %[[ARG4:.*]]: tensor<i64>)  {
// CHECK:         %[[T9:.*]] = mhlo.compare  GE, %[[ARG1]], %[[ARG3]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:         %[[T10:.*]] = mhlo.select %[[T9]], %[[ARG1]], %[[ARG3]] : tensor<i1>, tensor<f32>
// CHECK:         %[[T11:.*]] = mhlo.compare  EQ, %[[ARG1]], %[[ARG3]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:         %[[T12:.*]] = mhlo.minimum %[[ARG2]], %[[ARG4]] : tensor<i64>
// CHECK:         %[[T13:.*]] = mhlo.select %[[T9]], %[[ARG2]], %[[ARG4]] : tensor<i1>, tensor<i64>
// CHECK:         %[[T14:.*]] = mhlo.select %[[T11]], %[[T12]], %[[T13]] : tensor<i1>, tensor<i64>
// CHECK:         mhlo.return %[[T10]], %[[T14]] : tensor<f32>, tensor<i64>
// CHECK:         %[[T7:.*]] = torch_c.from_builtin_tensor %[[T6]]#0 : tensor<?xf32> -> !torch.vtensor<[?],f32>
// CHECK:         %[[T8:.*]] = torch_c.from_builtin_tensor %[[T6]]#1 : tensor<?xi64> -> !torch.vtensor<[?],si64>
// CHECK:         return %[[T7]], %[[T8]] : !torch.vtensor<[?],f32>, !torch.vtensor<[?],si64>
func.func @torch.aten.max.dim(%arg0: !torch.vtensor<[?,?],f32>) -> (!torch.vtensor<[?],f32>, !torch.vtensor<[?],si64>) {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %values, %indices = torch.aten.max.dim %arg0, %int1, %false : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?],f32>, !torch.vtensor<[?],si64>
  return %values, %indices : !torch.vtensor<[?],f32>, !torch.vtensor<[?],si64>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.argmax$keepdim(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,1],si64> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[TRUE:.*]] = torch.constant.bool true
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM_0:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[DIM_0]] : index to i64
// CHECK:         %[[T3:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:         %[[T4:.*]] = mhlo.constant dense<0> : tensor<i64>
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[T1]], %[[T2]] : tensor<2xi64>
// CHECK:         %[[T5:.*]] = "mhlo.dynamic_iota"(%[[FROM_ELEMENTS]]) {iota_dimension = 1 : i64} : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:         %[[T6:.*]]:2 = mhlo.reduce(%[[T0]] init: %[[T3]]), (%[[T5]] init: %[[T4]]) across dimensions = [1] : (tensor<?x?xf32>, tensor<?x?xi64>, tensor<f32>, tensor<i64>) -> (tensor<?xf32>, tensor<?xi64>)
// CHECK:         reducer(%[[ARG1:.*]]: tensor<f32>, %[[ARG3:.*]]: tensor<f32>) (%[[ARG2:.*]]: tensor<i64>, %[[ARG4:.*]]: tensor<i64>)  {
// CHECK:         %[[T9:.*]] = mhlo.compare  GE, %[[ARG1]], %[[ARG3]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:         %[[T10:.*]] = mhlo.select %[[T9]], %[[ARG1]], %[[ARG3]] : tensor<i1>, tensor<f32>
// CHECK:         %[[T11:.*]] = mhlo.compare  EQ, %[[ARG1]], %[[ARG3]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:         %[[T12:.*]] = mhlo.minimum %[[ARG2]], %[[ARG4]] : tensor<i64>
// CHECK:         %[[T13:.*]] = mhlo.select %[[T9]], %[[ARG2]], %[[ARG4]] : tensor<i1>, tensor<i64>
// CHECK:         %[[T14:.*]] = mhlo.select %[[T11]], %[[T12]], %[[T13]] : tensor<i1>, tensor<i64>
// CHECK:         mhlo.return %[[T10]], %[[T14]] : tensor<f32>, tensor<i64>
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[FROM_ELEMENTS_1:.*]] = tensor.from_elements %[[T1]], %[[C1_I64]] : tensor<2xi64>
// CHECK:         %[[T7:.*]] = mhlo.dynamic_reshape %[[T6]]#1, %[[FROM_ELEMENTS_1]] : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
// CHECK:         %[[T8:.*]] = torch_c.from_builtin_tensor %[[T7]] : tensor<?x1xi64> -> !torch.vtensor<[?,1],si64>
// CHECK:         return %[[T8]] : !torch.vtensor<[?,1],si64>
func.func @torch.aten.argmax$keepdim(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,1],si64> {
  %int1 = torch.constant.int 1
  %true = torch.constant.bool true
  %indices = torch.aten.argmax %arg0, %int1, %true : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?,1],si64>
  return %indices : !torch.vtensor<[?,1],si64>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.argmax(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?],si64> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[FALSE:.*]] = torch.constant.bool false
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf32>
// CHECK:         %[[T1:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM_0:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:         %[[T2:.*]] = arith.index_cast %[[DIM_0]] : index to i64
// CHECK:         %[[T3:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:         %[[T4:.*]] = mhlo.constant dense<0> : tensor<i64>
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[T1]], %[[T2]] : tensor<2xi64>
// CHECK:         %[[T5:.*]] = "mhlo.dynamic_iota"(%[[FROM_ELEMENTS]]) {iota_dimension = 1 : i64} : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:         %[[T6:.*]]:2 = mhlo.reduce(%[[T0]] init: %[[T3]]), (%[[T5]] init: %[[T4]]) across dimensions = [1] : (tensor<?x?xf32>, tensor<?x?xi64>, tensor<f32>, tensor<i64>) -> (tensor<?xf32>, tensor<?xi64>)
// CHECK:         reducer(%[[ARG1:.*]]: tensor<f32>, %[[ARG3:.*]]: tensor<f32>) (%[[ARG2:.*]]: tensor<i64>, %[[ARG4:.*]]: tensor<i64>)  {
// CHECK:         %[[T8:.*]] = mhlo.compare  GE, %[[ARG1]], %[[ARG3]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:         %[[T9:.*]] = mhlo.select %[[T8]], %[[ARG1]], %[[ARG3]] : tensor<i1>, tensor<f32>
// CHECK:         %[[T10:.*]] = mhlo.compare  EQ, %[[ARG1]], %[[ARG3]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:         %[[T11:.*]] = mhlo.minimum %[[ARG2]], %[[ARG4]] : tensor<i64>
// CHECK:         %[[T12:.*]] = mhlo.select %[[T8]], %[[ARG2]], %[[ARG4]] : tensor<i1>, tensor<i64>
// CHECK:         %[[T13:.*]] = mhlo.select %[[T10]], %[[T11]], %[[T12]] : tensor<i1>, tensor<i64>
// CHECK:         mhlo.return %[[T9]], %[[T13]] : tensor<f32>, tensor<i64>
// CHECK:         %[[T7:.*]] = torch_c.from_builtin_tensor %[[T6]]#1 : tensor<?xi64> -> !torch.vtensor<[?],si64>
// CHECK:         return %[[T7]] : !torch.vtensor<[?],si64>
func.func @torch.aten.argmax(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?],si64> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %indices = torch.aten.argmax %arg0, %int1, %false : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?],si64>
  return %indices : !torch.vtensor<[?],si64>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.sum.dim_Intlist$keepdim(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[1,1,?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[TRUE:.*]] = torch.constant.bool true
// CHECK:         %[[NONE:.*]] = torch.constant.none
// CHECK:         %[[T1:.*]] = torch.prim.ListConstruct %[[INT0]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[T2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T3:.*]] = mhlo.reduce(%[[T0]] init: %[[T2]]) applies mhlo.add across dimensions = [0, 1] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?x?xf32>
// CHECK:         %[[T4:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM_0:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?x?xf32>
// CHECK:         %[[T5:.*]] = arith.index_cast %[[DIM_0]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[DIM_1:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x?x?xf32>
// CHECK:         %[[T6:.*]] = arith.index_cast %[[DIM_1]] : index to i64
// CHECK:         %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[C1_I64]], %[[C1_I64]], %[[T6]] : tensor<3xi64>
// CHECK:         %[[T7:.*]] = mhlo.dynamic_reshape %[[T3]], %[[FROM_ELEMENTS]] : (tensor<?xf32>, tensor<3xi64>) -> tensor<1x1x?xf32>
// CHECK:         %[[T8:.*]] = torch_c.from_builtin_tensor %[[T7]] : tensor<1x1x?xf32> -> !torch.vtensor<[1,1,?],f32>
// CHECK:         return %[[T8]] : !torch.vtensor<[1,1,?],f32>
func.func @torch.aten.sum.dim_Intlist$keepdim(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[1,1,?],f32> {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %true = torch.constant.bool true
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.sum.dim_IntList %arg0, %0, %true, %none : !torch.vtensor<[?,?,?],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,?],f32>
  return %1 : !torch.vtensor<[1,1,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.sum.dim_Intlist(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[FALSE:.*]] = torch.constant.bool false
// CHECK:         %[[NONE:.*]] = torch.constant.none
// CHECK:         %[[T1:.*]] = torch.prim.ListConstruct %[[INT0]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[T2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T3:.*]] = mhlo.reduce(%[[T0]] init: %[[T2]]) applies mhlo.add across dimensions = [0, 1] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?xf32>
// CHECK:         %[[T4:.*]] = torch_c.from_builtin_tensor %[[T3]] : tensor<?xf32> -> !torch.vtensor<[?],f32>
// CHECK:         return %[[T4]] : !torch.vtensor<[?],f32>
func.func @torch.aten.sum.dim_Intlist(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?],f32> {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.sum.dim_IntList %arg0, %0, %false, %none : !torch.vtensor<[?,?,?],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[?],f32>
  return %1 : !torch.vtensor<[?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.sum(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:         %[[NONE:.*]] = torch.constant.none
// CHECK:         %[[T1:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:         %[[T2:.*]] = mhlo.reduce(%[[T0]] init: %[[T1]]) applies mhlo.add across dimensions = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<f32>
// CHECK:         %[[T3:.*]] = torch_c.from_builtin_tensor %[[T2]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:         return %[[T3]] : !torch.vtensor<[],f32>
func.func @torch.aten.sum(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
  %none = torch.constant.none
  %0 = torch.aten.sum %arg0, %none : !torch.vtensor<[?,?,?],f32>, !torch.none -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.max(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:         %[[T1:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:         %[[T2:.*]] = mhlo.reduce(%[[T0]] init: %[[T1]]) applies mhlo.maximum across dimensions = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<f32>
// CHECK:         %[[T3:.*]] = torch_c.from_builtin_tensor %[[T2]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:         return %[[T3]] : !torch.vtensor<[],f32>
func.func @torch.aten.max(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
  %0 = torch.aten.max %arg0 : !torch.vtensor<[?,?,?],f32> -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

