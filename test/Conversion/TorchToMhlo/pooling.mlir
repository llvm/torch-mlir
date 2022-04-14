// RUN: torch-mlir-opt <%s -convert-torch-to-mhlo -split-input-file -verify-diagnostics | FileCheck %s

// -----
// CHECK-LABEL:   func.func @torch.aten.max_pool2d(
// CHECK-SAME:                                %[[VAL_0:.*]]: !torch.vtensor<[2,5,4,4],f32>) -> !torch.vtensor<[2,5,2,3],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,5,4,4],f32> -> tensor<2x5x4x4xf32>
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
// CHECK:             "mhlo.return"(%[[VAL_10]]) : (tensor<f32>) -> ()
// CHECK:           }) {padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<[1, 1, 2, 1]> : tensor<4xi64>, window_dimensions = dense<[1, 1, 2, 2]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<2x5x4x4xf32>, tensor<f32>) -> tensor<2x5x2x3xf32>
// CHECK:           %[[VAL_11:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<2x5x2x3xf32> -> !torch.vtensor<[2,5,2,3],f32>
// CHECK:           return %[[VAL_11]] : !torch.vtensor<[2,5,2,3],f32>
func.func @torch.aten.max_pool2d(%arg0: !torch.vtensor<[2,5,4,4],f32>) -> !torch.vtensor<[2,5,2,3],f32> {
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.aten.max_pool2d %arg0, %0, %1, %2, %3, %false : !torch.vtensor<[2,5,4,4],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,5,2,3],f32>
  return %4 : !torch.vtensor<[2,5,2,3],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.max_pool2d$padding(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[2,5,4,4],f32>) -> !torch.vtensor<[2,5,6,5],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,5,4,4],f32> -> tensor<2x5x4x4xf32>
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
// CHECK:             "mhlo.return"(%[[VAL_10]]) : (tensor<f32>) -> ()
// CHECK:           }) 
// CHECK-SAME{LITERAL}:  {padding = dense<[[0, 0], [0, 0], [2, 2], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<[1, 1, 2, 1]> : tensor<4xi64>, window_dimensions = dense<[1, 1, 2, 2]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<2x5x4x4xf32>, tensor<f32>) -> tensor<2x5x6x5xf32>
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]] : tensor<2x5x6x5xf32> -> !torch.vtensor<[2,5,6,5],f32>
// CHECK:           return %[[VAL_7]] : !torch.vtensor<[2,5,6,5],f32>
func.func @torch.aten.max_pool2d$padding(%arg0: !torch.vtensor<[2,5,4,4],f32>) -> !torch.vtensor<[2,5,6,5],f32> {
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.max_pool2d %arg0, %0, %1, %2, %2, %false : !torch.vtensor<[2,5,4,4],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,5,6,5],f32>
  return %3 : !torch.vtensor<[2,5,6,5],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.adaptive_avg_pool2d(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !torch.vtensor<[2,5,7],f32>) -> !torch.vtensor<[2,1,1],f32> {
// CEHCK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,5,7],f32> -> tensor<2x5x7xf32>
// CEHCK:           %int1 = torch.constant.int 1
// CEHCK:           %[[VAL_2:.*]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CEHCK:           %[[VAL_3:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CEHCK:           %[[VAL_4:.*]] = "mhlo.reduce_window"(%[[VAL_1]], %[[VAL_3]]) ({
// CEHCK:           ^bb0(%[[VAL_9:.*]]: tensor<f32>, %[[VAL_10:.*]]: tensor<f32>):
// CEHCK:             %[[VAL_11:.*]] = mhlo.add %[[VAL_9]], %[[VAL_10]] : tensor<f32>
// CEHCK:             "mhlo.return"(%[[VAL_11]]) : (tensor<f32>) -> ()
// CEHCK:           }) {window_dimensions = dense<[1, 5, 7]> : tensor<3xi64>} : (tensor<2x5x7xf32>, tensor<f32>) -> tensor<2x1x1xf32>
// CEHCK:           %[[VAL_5:.*]] = mhlo.constant dense<3.500000e+01> : tensor<f32>
// CEHCK:           %[[VAL_6:.*]] = "mhlo.broadcast_in_dim"(%[[VAL_5]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<2x1x1xf32>
// CEHCK:           %[[VAL_7:.*]] = mhlo.divide %[[VAL_4]], %[[VAL_6]] : tensor<2x1x1xf32>
// CEHCK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_7]] : tensor<2x1x1xf32> -> !torch.vtensor<[2,1,1],f32>
// CEHCK:           return %[[VAL_8]] : !torch.vtensor<[2,1,1],f32>
func.func @torch.aten.adaptive_avg_pool2d(%arg0: !torch.vtensor<[2,5,7],f32>) -> !torch.vtensor<[2,1,1],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.adaptive_avg_pool2d %arg0, %0 : !torch.vtensor<[2,5,7],f32>, !torch.list<int> -> !torch.vtensor<[2,1,1],f32>
  return %1 : !torch.vtensor<[2,1,1],f32>
}


// -----
// CHECK-LABEL:   func.func @torch.aten.max_pool2d_with_indices(
// CHECK-SAME:                                             %[[VAL_0:.*]]: !torch.vtensor<[5,5,5],f32>) -> (!torch.vtensor<[5,2,2],f32>, !torch.vtensor<[5,2,2],si64>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[5,5,5],f32> -> tensor<5x5x5xf32>
// CHECK:           %int3 = torch.constant.int 3
// CHECK:           %int2 = torch.constant.int 2
// CHECK:           %false = torch.constant.bool false
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %1 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %2 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %3 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %4 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = mhlo.constant dense<[5, 25]> : tensor<2xi64>
// CHECK:           %[[VAL_4:.*]] = mhlo.constant dense<5> : tensor<3xi64>
// CHECK:           %[[VAL_5:.*]] = "mhlo.dynamic_iota"(%[[VAL_3]]) {iota_dimension = 1 : i64} : (tensor<2xi64>) -> tensor<5x25xi64>
// CHECK:           %[[VAL_6:.*]] = "mhlo.dynamic_reshape"(%[[VAL_5]], %[[VAL_4]]) : (tensor<5x25xi64>, tensor<3xi64>) -> tensor<5x5x5xi64>
// CHECK:           %[[VAL_7:.*]] = mhlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_8:.*]]:2 = "mhlo.reduce_window"(%[[VAL_1]], %[[VAL_6]], %[[VAL_2]], %[[VAL_7]]) ({
// CHECK:           ^bb0(%[[VAL_11:.*]]: tensor<f32>, %[[VAL_12:.*]]: tensor<i64>, %[[VAL_13:.*]]: tensor<f32>, %[[VAL_14:.*]]: tensor<i64>):
// CHECK:             %[[VAL_15:.*]] = "mhlo.compare"(%[[VAL_11]], %[[VAL_13]]) {compare_type = #mhlo<"comparison_type FLOAT">, comparison_direction = #mhlo<"comparison_direction GE">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             %[[VAL_16:.*]] = "mhlo.select"(%[[VAL_15]], %[[VAL_11]], %[[VAL_13]]) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:             %[[VAL_17:.*]] = "mhlo.compare"(%[[VAL_11]], %[[VAL_13]]) {compare_type = #mhlo<"comparison_type FLOAT">, comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             %[[VAL_18:.*]] = mhlo.minimum %[[VAL_12]], %[[VAL_14]] : tensor<i64>
// CHECK:             %[[VAL_19:.*]] = "mhlo.select"(%[[VAL_15]], %[[VAL_12]], %[[VAL_14]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:             %[[VAL_20:.*]] = "mhlo.select"(%[[VAL_17]], %[[VAL_18]], %[[VAL_19]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:             "mhlo.return"(%[[VAL_16]], %[[VAL_20]]) : (tensor<f32>, tensor<i64>) -> ()
// CHECK:           }) {padding = dense<0> : tensor<3x2xi64>, window_dilations = dense<1> : tensor<3xi64>, window_dimensions = dense<[1, 3, 3]> : tensor<3xi64>, window_strides = dense<[1, 2, 2]> : tensor<3xi64>} : (tensor<5x5x5xf32>, tensor<5x5x5xi64>, tensor<f32>, tensor<i64>) -> (tensor<5x2x2xf32>, tensor<5x2x2xi64>)
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]]#0 : tensor<5x2x2xf32> -> !torch.vtensor<[5,2,2],f32>
// CHECK:           %[[VAL_10:.*]] = torch_c.from_builtin_tensor %[[VAL_8]]#1 : tensor<5x2x2xi64> -> !torch.vtensor<[5,2,2],si64>
// CHECK:           return %[[VAL_9]], %[[VAL_10]] : !torch.vtensor<[5,2,2],f32>, !torch.vtensor<[5,2,2],si64>
func.func @torch.aten.max_pool2d_with_indices(%arg0: !torch.vtensor<[5,5,5],f32>) -> (!torch.vtensor<[5,2,2],f32>, !torch.vtensor<[5,2,2],si64>) {
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %result0, %result1 = torch.aten.max_pool2d_with_indices %arg0, %0, %1, %2, %3, %false : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,2,2],f32>, !torch.vtensor<[5,2,2],si64>
  return %result0, %result1 : !torch.vtensor<[5,2,2],f32>, !torch.vtensor<[5,2,2],si64>
}