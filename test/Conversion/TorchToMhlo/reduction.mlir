// RUN: torch-mlir-opt <%s -convert-torch-to-mhlo -split-input-file -verify-diagnostics | FileCheck %s

// -----
// CHECK-LABEL:   func.func @torch.aten.max.dim$keepdim(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[3,5],f32>) -> (!torch.vtensor<[1,5],f32>, !torch.vtensor<[1,5],si64>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,5],f32> -> tensor<3x5xf32>
// CHECK:           %true = torch.constant.bool true
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = mhlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_4:.*]] = mhlo.constant dense<[3, 5]> : tensor<2xi64>
// CHECK:           %[[VAL_5:.*]] = "mhlo.dynamic_iota"(%[[VAL_4]]) {iota_dimension = 0 : i64} : (tensor<2xi64>) -> tensor<3x5xi64>
// CHECK:           %[[VAL_6:.*]]:2 = mhlo.reduce(%[[VAL_1]] init: %[[VAL_2]]), (%[[VAL_5]] init: %[[VAL_3:.*]]) across dimensions = [0] : (tensor<3x5xf32>, tensor<3x5xi64>, tensor<f32>, tensor<i64>) -> (tensor<5xf32>, tensor<5xi64>)
// CHECK:             reducer(%[[VAL_12:.*]]: tensor<f32>, %[[VAL_13:.*]]: tensor<f32>) (%[[VAL_14:.*]]: tensor<i64>, %[[VAL_15:.*]]: tensor<i64>)  {
// CHECK:             %[[VAL_16:.*]] = "mhlo.compare"(%[[VAL_12]], %[[VAL_13]]) {compare_type = #mhlo<"comparison_type FLOAT">, comparison_direction = #mhlo<"comparison_direction GE">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             %[[VAL_17:.*]] = "mhlo.select"(%[[VAL_16]], %[[VAL_12]], %[[VAL_13]]) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:             %[[VAL_18:.*]] = "mhlo.compare"(%[[VAL_12]], %[[VAL_13]]) {compare_type = #mhlo<"comparison_type FLOAT">, comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             %[[VAL_19:.*]] = mhlo.minimum %[[VAL_14]], %[[VAL_15]] : tensor<i64>
// CHECK:             %[[VAL_20:.*]] = "mhlo.select"(%[[VAL_16]], %[[VAL_14]], %[[VAL_15]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:             %[[VAL_21:.*]] = "mhlo.select"(%[[VAL_18]], %[[VAL_19]], %[[VAL_20]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:             "mhlo.return"(%[[VAL_17]], %[[VAL_21]]) : (tensor<f32>, tensor<i64>) -> ()
// CHECK:           }
// CHECK:           %[[VAL_7:.*]] = mhlo.constant dense<[1, 5]> : tensor<2xi64>
// CHECK:           %[[VAL_8:.*]] = "mhlo.dynamic_reshape"(%[[VAL_6]]#0, %[[VAL_7]]) : (tensor<5xf32>, tensor<2xi64>) -> tensor<1x5xf32>
// CHECK:           %[[VAL_9:.*]] = "mhlo.dynamic_reshape"(%[[VAL_6]]#1, %[[VAL_7]]) : (tensor<5xi64>, tensor<2xi64>) -> tensor<1x5xi64>
// CHECK:           %[[VAL_10:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<1x5xf32> -> !torch.vtensor<[1,5],f32>
// CHECK:           %[[VAL_11:.*]] = torch_c.from_builtin_tensor %[[VAL_9]] : tensor<1x5xi64> -> !torch.vtensor<[1,5],si64>
// CHECK:           return %[[VAL_10]], %[[VAL_11]] : !torch.vtensor<[1,5],f32>, !torch.vtensor<[1,5],si64>
func.func @torch.aten.max.dim$keepdim(%arg0: !torch.vtensor<[3,5],f32>) -> (!torch.vtensor<[1,5],f32>, !torch.vtensor<[1,5],si64>) {
  %true = torch.constant.bool true
  %int0 = torch.constant.int 0
  %values, %indices = torch.aten.max.dim %arg0, %int0, %true : !torch.vtensor<[3,5],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,5],f32>, !torch.vtensor<[1,5],si64>
  return %values, %indices : !torch.vtensor<[1,5],f32>, !torch.vtensor<[1,5],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.max.dim(
// CHECK-SAME:                             %[[VAL_0:.*]]: !torch.vtensor<[3,5],f64>) -> (!torch.vtensor<[5],f64>, !torch.vtensor<[5],si64>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,5],f64> -> tensor<3x5xf64>
// CHECK:           %false = torch.constant.bool false
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<-1.7976931348623157E+308> : tensor<f64>
// CHECK:           %[[VAL_3:.*]] = mhlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_4:.*]] = mhlo.constant dense<[3, 5]> : tensor<2xi64>
// CHECK:           %[[VAL_5:.*]] = "mhlo.dynamic_iota"(%[[VAL_4]]) {iota_dimension = 0 : i64} : (tensor<2xi64>) -> tensor<3x5xi64>
// CHECK:           %[[VAL_6:.*]]:2 = mhlo.reduce(%[[VAL_1]] init: %[[VAL_2]]), (%[[VAL_5]] init: %[[VAL_3]]) across dimensions = [0] : (tensor<3x5xf64>, tensor<3x5xi64>, tensor<f64>, tensor<i64>) -> (tensor<5xf64>, tensor<5xi64>)
// CHECK:             reducer(%[[VAL_9:.*]]: tensor<f64>, %[[VAL_10:.*]]: tensor<f64>) (%[[VAL_11:.*]]: tensor<i64>, %[[VAL_12:.*]]: tensor<i64>)  {
// CHECK:             %[[VAL_13:.*]] = "mhlo.compare"(%[[VAL_9]], %[[VAL_10]]) {compare_type = #mhlo<"comparison_type FLOAT">, comparison_direction = #mhlo<"comparison_direction GE">} : (tensor<f64>, tensor<f64>) -> tensor<i1>
// CHECK:             %[[VAL_14:.*]] = "mhlo.select"(%[[VAL_13]], %[[VAL_9]], %[[VAL_10]]) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK:             %[[VAL_15:.*]] = "mhlo.compare"(%[[VAL_9]], %[[VAL_10]]) {compare_type = #mhlo<"comparison_type FLOAT">, comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<f64>, tensor<f64>) -> tensor<i1>
// CHECK:             %[[VAL_16:.*]] = mhlo.minimum %[[VAL_11]], %[[VAL_12]] : tensor<i64>
// CHECK:             %[[VAL_17:.*]] = "mhlo.select"(%[[VAL_13]], %[[VAL_11]], %[[VAL_12]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:             %[[VAL_18:.*]] = "mhlo.select"(%[[VAL_15]], %[[VAL_16]], %[[VAL_17]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:             "mhlo.return"(%[[VAL_14]], %[[VAL_18]]) : (tensor<f64>, tensor<i64>) -> ()
// CHECK:           }
// CHECK:           %[[VAL_7:.*]] = torch_c.from_builtin_tensor %[[VAL_6]]#0 : tensor<5xf64> -> !torch.vtensor<[5],f64>
// CHECK:           %[[VAL_8:.*]] = torch_c.from_builtin_tensor %[[VAL_6]]#1 : tensor<5xi64> -> !torch.vtensor<[5],si64>
// CHECK:           return %[[VAL_7]], %[[VAL_8]] : !torch.vtensor<[5],f64>, !torch.vtensor<[5],si64>
func.func @torch.aten.max.dim(%arg0: !torch.vtensor<[3,5],f64>) -> (!torch.vtensor<[5],f64>, !torch.vtensor<[5],si64>) {
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %values, %indices = torch.aten.max.dim %arg0, %int0, %false : !torch.vtensor<[3,5],f64>, !torch.int, !torch.bool -> !torch.vtensor<[5],f64>, !torch.vtensor<[5],si64>
  return %values, %indices : !torch.vtensor<[5],f64>, !torch.vtensor<[5],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.sum.dim_Intlist(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[3,5,7],f32>) -> !torch.vtensor<[1,1,7],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,5,7],f32> -> tensor<3x5x7xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 0
// CHECK:           %[[VAL_4:.*]] = torch.constant.bool true
// CHECK:           %[[VAL_5:.*]] = torch.constant.none
// CHECK:           %[[VAL_6:.*]] = torch.prim.ListConstruct %[[VAL_3]], %[[VAL_2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_7:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_8:.*]] = mhlo.reduce(%[[VAL_1]] init: %[[VAL_7]]) applies mhlo.add across dimensions = [0, 1] : (tensor<3x5x7xf32>, tensor<f32>) -> tensor<7xf32>
// CHECK:           %[[VAL_9:.*]] = mhlo.constant dense<[1, 1, 7]> : tensor<3xi64>
// CHECK:           %[[VAL_10:.*]] = "mhlo.dynamic_reshape"(%[[VAL_8]], %[[VAL_9]]) : (tensor<7xf32>, tensor<3xi64>) -> tensor<1x1x7xf32>
// CHECK:           %[[VAL_11:.*]] = torch_c.from_builtin_tensor %[[VAL_10]] : tensor<1x1x7xf32> -> !torch.vtensor<[1,1,7],f32>
// CHECK:           return %[[VAL_11]] : !torch.vtensor<[1,1,7],f32>
func.func @torch.aten.sum.dim_Intlist(%arg0: !torch.vtensor<[3,5,7],f32>) -> !torch.vtensor<[1,1,7],f32> {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %true = torch.constant.bool true
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.sum.dim_IntList %arg0, %0, %true, %none : !torch.vtensor<[3,5,7],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,7],f32>
  return %1 : !torch.vtensor<[1,1,7],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.sum.dim_Intlist$convert(
// CHECK-SAME:                                             %[[VAL_0:.*]]: !torch.vtensor<[3,5,7],f16>) -> !torch.vtensor<[7],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,5,7],f16> -> tensor<3x5x7xf16>
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %false = torch.constant.bool false
// CHECK:           %int6 = torch.constant.int 6
// CHECK:           %[[VAL_2:.*]] = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_3:.*]] = mhlo.convert(%[[VAL_1]]) : (tensor<3x5x7xf16>) -> tensor<3x5x7xf32>
// CHECK:           %[[VAL_4:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_5:.*]] = mhlo.reduce(%[[VAL_3]] init: %[[VAL_4]]) applies mhlo.add across dimensions = [0, 1] : (tensor<3x5x7xf32>, tensor<f32>) -> tensor<7xf32>
// CHECK:           %[[VAL_6:.*]] = torch_c.from_builtin_tensor %[[VAL_5]] : tensor<7xf32> -> !torch.vtensor<[7],f32>
// CHECK:           return %[[VAL_6]] : !torch.vtensor<[7],f32>
func.func @torch.aten.sum.dim_Intlist$convert(%arg0: !torch.vtensor<[3,5,7],f16>) -> !torch.vtensor<[7],f32> {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %int6 = torch.constant.int 6
  %0 = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.sum.dim_IntList %arg0, %0, %false, %int6 : !torch.vtensor<[3,5,7],f16>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[7],f32>
  return %1 : !torch.vtensor<[7],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.argmax$keepdim(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !torch.vtensor<[3,5,7],f32>) -> !torch.vtensor<[3,1,7],si64> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,5,7],f32> -> tensor<3x5x7xf32>
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %true = torch.constant.bool true
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = mhlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_4:.*]] = mhlo.constant dense<[3, 5, 7]> : tensor<3xi64>
// CHECK:           %[[VAL_5:.*]] = "mhlo.dynamic_iota"(%[[VAL_4]]) {iota_dimension = 1 : i64} : (tensor<3xi64>) -> tensor<3x5x7xi64>
// CHECK:           %[[VAL_6:.*]]:2 = mhlo.reduce(%[[VAL_1]] init: %[[VAL_2]]), (%[[VAL_5]] init: %[[VAL_3]]) across dimensions = [1] : (tensor<3x5x7xf32>, tensor<3x5x7xi64>, tensor<f32>, tensor<i64>) -> (tensor<3x7xf32>, tensor<3x7xi64>)
// CHECK:             reducer(%[[VAL_10:.*]]: tensor<f32>, %[[VAL_11:.*]]: tensor<f32>) (%[[VAL_12:.*]]: tensor<i64>, %[[VAL_13:.*]]: tensor<i64>)  {
// CHECK:             %[[VAL_14:.*]] = "mhlo.compare"(%[[VAL_10]], %[[VAL_11]]) {compare_type = #mhlo<"comparison_type FLOAT">, comparison_direction = #mhlo<"comparison_direction GE">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             %[[VAL_15:.*]] = "mhlo.select"(%[[VAL_14]], %[[VAL_10]], %[[VAL_11]]) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:             %[[VAL_16:.*]] = "mhlo.compare"(%[[VAL_10]], %[[VAL_11]]) {compare_type = #mhlo<"comparison_type FLOAT">, comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             %[[VAL_17:.*]] = mhlo.minimum %[[VAL_12]], %[[VAL_13]] : tensor<i64>
// CHECK:             %[[VAL_18:.*]] = "mhlo.select"(%[[VAL_14]], %[[VAL_12]], %[[VAL_13]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:             %[[VAL_19:.*]] = "mhlo.select"(%[[VAL_16]], %[[VAL_17]], %[[VAL_18]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:             "mhlo.return"(%[[VAL_15]], %[[VAL_19]]) : (tensor<f32>, tensor<i64>) -> ()
// CHECK:           }
// CHECK:           %[[VAL_7:.*]] = mhlo.constant dense<[3, 1, 7]> : tensor<3xi64>
// CHECK:           %[[VAL_8:.*]] = "mhlo.dynamic_reshape"(%[[VAL_6]]#1, %[[VAL_7]]) : (tensor<3x7xi64>, tensor<3xi64>) -> tensor<3x1x7xi64>
// CHECK:           %[[VAL_9:.*]] = torch_c.from_builtin_tensor %[[VAL_8]] : tensor<3x1x7xi64> -> !torch.vtensor<[3,1,7],si64>
// CHECK:           return %[[VAL_9]] : !torch.vtensor<[3,1,7],si64>
func.func @torch.aten.argmax$keepdim(%arg0: !torch.vtensor<[3,5,7],f32>) -> !torch.vtensor<[3,1,7],si64> {
  %int1 = torch.constant.int 1
  %true = torch.constant.bool true
  %values, %indices = torch.aten.max.dim %arg0, %int1, %true : !torch.vtensor<[3,5,7],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3,1,7],f32>, !torch.vtensor<[3,1,7],si64>
  return %indices : !torch.vtensor<[3,1,7],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.sum(
// CHECK-SAME:                         %[[VAL_0:.*]]: !torch.vtensor<[3,5,7],f32>) -> !torch.vtensor<[],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,5,7],f32> -> tensor<3x5x7xf32>
// CHECK:           %none = torch.constant.none
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = mhlo.reduce(%[[VAL_1]] init: %[[VAL_2]]) applies mhlo.add across dimensions = [0, 1, 2] : (tensor<3x5x7xf32>, tensor<f32>) -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[],f32>
func.func @torch.aten.sum(%arg0: !torch.vtensor<[3,5,7],f32>) -> !torch.vtensor<[],f32> {
  %none = torch.constant.none
  %0 = torch.aten.sum %arg0, %none : !torch.vtensor<[3,5,7],f32>, !torch.none -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.max(
// CHECK-SAME:                         %[[VAL_0:.*]]: !torch.vtensor<[3,5,7],f32>) -> !torch.vtensor<[],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[3,5,7],f32> -> tensor<3x5x7xf32>
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = mhlo.reduce(%[[VAL_1]] init: %[[VAL_2]]) applies mhlo.maximum across dimensions = [0, 1, 2] : (tensor<3x5x7xf32>, tensor<f32>) -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[],f32>
func.func @torch.aten.max(%arg0: !torch.vtensor<[3,5,7],f32>) -> !torch.vtensor<[],f32> {
  %0 = torch.aten.max %arg0 : !torch.vtensor<[3,5,7],f32> -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}