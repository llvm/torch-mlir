// RUN: torch-mlir-opt <%s -convert-torch-to-mhlo -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL:   func.func @torch.aten.max_pool2d(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %int2 = torch.constant.int 2
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %false = torch.constant.bool false
// CHECK:           %[[VAL_2:.*]] = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_3:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_6:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:           %[[VAL_7:.*]] = "mhlo.reduce_window"(%[[VAL_1]], %[[VAL_6]]) ({
// CHECK:           ^bb0(%[[VAL_8:.*]]: tensor<f32>, %[[VAL_9:.*]]: tensor<f32>):
// CHECK:             %[[VAL_10:.*]] = mhlo.maximum %[[VAL_8]], %[[VAL_9]] : tensor<f32>
// CHECK:             mhlo.return %[[VAL_10]] : tensor<f32>
// CHECK:           }) {padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<[1, 1, 2, 1]> : tensor<4xi64>, window_dimensions = dense<[1, 1, 2, 2]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_11:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[VAL_11]] : !torch.vtensor<[?,?,?,?],f32>
func.func @torch.aten.max_pool2d(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.aten.max_pool2d %arg0, %0, %1, %2, %3, %false : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  return %4 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.max_pool2d$padding(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %int2 = torch.constant.int 2
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %false = torch.constant.bool false
// CHECK:           %[[VAL_2:.*]] = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_3:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = "mhlo.reduce_window"(%[[VAL_1]], %[[VAL_5]]) ({
// CHECK:           ^bb0(%[[VAL_8:.*]]: tensor<f32>, %[[VAL_9:.*]]: tensor<f32>):
// CHECK:             %[[VAL_10:.*]] = mhlo.maximum %[[VAL_8]], %[[VAL_9]] : tensor<f32>
// CHECK:             mhlo.return %[[VAL_10]] : tensor<f32>
// CHECK:           }) 
// CHECK-SAME{LITERAL}:  {padding = dense<[[0, 0], [0, 0], [2, 2], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<[1, 1, 2, 1]> : tensor<4xi64>, window_dimensions = dense<[1, 1, 2, 2]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[?,?,?,?],f32>
func.func @torch.aten.max_pool2d$padding(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.max_pool2d %arg0, %0, %1, %2, %2, %false : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  return %3 : !torch.vtensor<[?,?,?,?],f32>
}


// -----

// CHECK-LABEL:  func.func @torch.aten.max_pool2d_with_indices(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?],f32>) -> (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],si64>) {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:         %[[INT3:.*]] = torch.constant.int 3
// CHECK:         %[[INT2:.*]] = torch.constant.int 2
// CHECK:         %[[FALSE:.*]] = torch.constant.bool false
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:         %[[T1:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[T2:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[T3:.*]] = torch.prim.ListConstruct %[[INT0]], %[[INT0]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[T4:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:         %[[T5:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?x?xf32>
// CHECK:         %[[T6:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM_0:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?x?xf32>
// CHECK:         %[[T7:.*]] = arith.index_cast %[[DIM_0]] : index to i64
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[DIM_1:.*]] = tensor.dim %[[T0]], %[[C2]] : tensor<?x?x?xf32>
// CHECK:         %[[T8:.*]] = arith.index_cast %[[DIM_1]] : index to i64
// CHECK:         %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[T6]], %[[T7]], %[[T8]] : tensor<3xi64>
// CHECK:         %[[T9:.*]] = arith.muli %[[T8]], %[[T7]] : i64
// CHECK:         %[[FROM_ELEMENTS_2:.*]] = tensor.from_elements %[[T6]], %[[T9]] : tensor<2xi64>
// CHECK:         %[[T10:.*]] = "mhlo.dynamic_iota"(%[[FROM_ELEMENTS_2]]) {iota_dimension = 1 : i64} : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:         %[[T11:.*]] = mhlo.dynamic_reshape %[[T10]], %[[FROM_ELEMENTS]] : (tensor<?x?xi64>, tensor<3xi64>) -> tensor<?x?x?xi64>
// CHECK:         %[[T12:.*]] = mhlo.constant dense<0> : tensor<i64>
// CHECK:         %[[T13:.*]]:2 = "mhlo.reduce_window"(%[[T0]], %[[T11]], %[[T5]], %[[T12]]) ({
// CHECK:         ^bb0(%[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<i64>, %[[ARG3:.*]]: tensor<f32>, %[[ARG4:.*]]: tensor<i64>):
// CHECK:         %[[T16:.*]] = mhlo.compare  GE, %[[ARG1]], %[[ARG3]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:         %[[T17:.*]] = mhlo.select %[[T16]], %[[ARG1]], %[[ARG3]] : tensor<i1>, tensor<f32>
// CHECK:         %[[T18:.*]] = mhlo.compare  EQ, %[[ARG1]], %[[ARG3]],  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:         %[[T19:.*]] = mhlo.minimum %[[ARG2]], %[[ARG4]] : tensor<i64>
// CHECK:         %[[T20:.*]] = mhlo.select %[[T16]], %[[ARG2]], %[[ARG4]] : tensor<i1>, tensor<i64>
// CHECK:         %[[T21:.*]] = mhlo.select %[[T18]], %[[T19]], %[[T20]] : tensor<i1>, tensor<i64>
// CHECK:         mhlo.return %[[T17]], %[[T21]] : tensor<f32>, tensor<i64>
// CHECK:         }) {padding = dense<0> : tensor<3x2xi64>, window_dilations = dense<1> : tensor<3xi64>, window_dimensions = dense<[1, 3, 3]> : tensor<3xi64>, window_strides = dense<[1, 2, 2]> : tensor<3xi64>} : (tensor<?x?x?xf32>, tensor<?x?x?xi64>, tensor<f32>, tensor<i64>) -> (tensor<?x?x?xf32>, tensor<?x?x?xi64>)
// CHECK:         %[[T14:.*]] = torch_c.from_builtin_tensor %[[T13]]#0 : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:         %[[T15:.*]] = torch_c.from_builtin_tensor %[[T13]]#1 : tensor<?x?x?xi64> -> !torch.vtensor<[?,?,?],si64>
// CHECK:         return %[[T14]], %[[T15]] : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],si64>
func.func @torch.aten.max_pool2d_with_indices(%arg0: !torch.vtensor<[?,?,?],f32>) -> (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],si64>) {
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %result0, %result1 = torch.aten.max_pool2d_with_indices %arg0, %0, %1, %2, %3, %false : !torch.vtensor<[?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],si64>
  return %result0, %result1 : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.avg_pool2d(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %int3 = torch.constant.int 3
// CHECK:           %int2 = torch.constant.int 2
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %false = torch.constant.bool false
// CHECK:           %none = torch.constant.none
// CHECK:           %[[VAL_2:.*]] = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_3:.*]] = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_4:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_5:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = "mhlo.reduce_window"(%[[VAL_1]], %[[VAL_5]]) ({
// CHECK:           ^bb0(%[[IVAL_0:.*]]: tensor<f32>, %[[IVAL_1:.*]]: tensor<f32>):
// CHECK:             %[[IVAL_2:.*]] = mhlo.add %[[IVAL_0]], %[[IVAL_1]] : tensor<f32>
// CHECK:             mhlo.return %[[IVAL_2]] : tensor<f32>
// CHECK{LITERAL}:  }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_7:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK:           %[[IDX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = tensor.dim %[[VAL_1]], %[[IDX_0]] : tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_9:.*]] = arith.index_cast %[[VAL_8]] : index to i64
// CHECK:           %[[IDX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_10:.*]] = tensor.dim %[[VAL_1]], %[[IDX_1]] : tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_11:.*]] = arith.index_cast %[[VAL_10]] : index to i64
// CHECK:           %[[IDX_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_12:.*]] = tensor.dim %[[VAL_1]], %[[IDX_2]] : tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_13:.*]] = arith.index_cast %[[VAL_12]] : index to i64
// CHECK:           %[[IDX_3:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_14:.*]] = tensor.dim %[[VAL_1]], %[[IDX_3]] : tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_15:.*]] = arith.index_cast %[[VAL_14]] : index to i64
// CHECK:           %[[VAL_16:.*]] = tensor.from_elements %[[VAL_9]], %[[VAL_11]], %[[VAL_13]], %[[VAL_15]] : tensor<4xi64>
// CHECK:           %[[VAL_17:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[VAL_7]], %[[VAL_16]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_18:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_19:.*]] = "mhlo.reduce_window"(%[[VAL_17]], %[[VAL_18]]) ({
// CHECK:           ^bb0(%[[IVAL_3:.*]]: tensor<f32>, %[[IVAL_4:.*]]: tensor<f32>):
// CHECK:             %[[IVAL_5:.*]] = mhlo.add %[[IVAL_3]], %[[IVAL_4]] : tensor<f32>
// CHECK:             mhlo.return %[[IVAL_5]] : tensor<f32>
// CHECK{LITERAL}:  }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_20:.*]] = mhlo.divide %[[VAL_6]], %[[VAL_19]] : tensor<?x?x?x?xf32>
// CHECK:           %[[VAL_21:.*]] = torch_c.from_builtin_tensor %[[VAL_20]] : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[VAL_21]] : !torch.vtensor<[?,?,?,?],f32>
func.func @torch.aten.avg_pool2d(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.avg_pool2d %arg0, %0, %1, %2, %false, %false, %none : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?,?],f32>
  return %3 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.avg_pool2d$count_include_pad(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK:           %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?,?,?],f32> -> tensor<?x?x?x?xf32>
// CHECK:           %[[INT3:.*]] = torch.constant.int 3
// CHECK:           %[[INT2:.*]] = torch.constant.int 2
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[T1:.*]] = torch.prim.ListConstruct %[[INT3]], %[[INT3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T2:.*]] = torch.prim.ListConstruct %[[INT2]], %[[INT2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T3:.*]] = torch.prim.ListConstruct %[[INT1]], %[[INT1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[T4:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[T5:.*]] = "mhlo.reduce_window"(%[[T0]], %[[T4]]) ({
// CHECK:           ^bb0(%[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<f32>):
// CHECK:             %[[T10:.*]] = mhlo.add %[[ARG1]], %[[ARG2]] : tensor<f32>
// CHECK:             mhlo.return %[[T10]] : tensor<f32>
// CHECK{LITERAL}:  }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[T6:.*]] = mhlo.constant dense<9> : tensor<i64>
// CHECK:           %[[T7:.*]] = mhlo.convert %[[T6]] : (tensor<i64>) -> tensor<f32>
// CHECK:           %[[T8:.*]] = chlo.broadcast_divide %[[T5]], %[[T7]] : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:           %[[T9:.*]] = torch_c.from_builtin_tensor %[[T8]] : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK:           return %[[T9]] : !torch.vtensor<[?,?,?,?],f32>
func.func @torch.aten.avg_pool2d$count_include_pad(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.avg_pool2d %arg0, %0, %1, %2, %false, %true, %none : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?,?],f32>
  return %3 : !torch.vtensor<[?,?,?,?],f32>
}
