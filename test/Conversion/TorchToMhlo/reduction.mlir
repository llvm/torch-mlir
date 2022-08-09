// RUN: torch-mlir-opt <%s -convert-torch-to-mhlo -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.max.dim$keepdim(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> (!torch.vtensor<[?,1],f32>, !torch.vtensor<[?,1],si64>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %true = torch.constant.bool true
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %[[IDX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = tensor.dim %[[VAL_1]], %[[IDX_0]] : tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_2]] : index to i64
// CHECK:           %[[IDX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = tensor.dim %[[VAL_1]], %[[IDX_1]] : tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = arith.index_cast %[[VAL_4]] : index to i64
// CHECK:           %[[VAL_6:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:           %[[VAL_7:.*]] = mhlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_8:.*]] = tensor.from_elements %[[VAL_3]], %[[VAL_5]] : tensor<2xi64>
// CHECK:           %[[VAL_9:.*]] = "mhlo.dynamic_iota"(%[[VAL_8]]) {iota_dimension = 1 : i64} : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:           %[[VAL_10:.*]]:2 = mhlo.reduce(%[[VAL_1]] init: %[[VAL_6]]), (%[[VAL_9]] init: %[[VAL_7]]) across dimensions = [1] : (tensor<?x?xf32>, tensor<?x?xi64>, tensor<f32>, tensor<i64>) -> (tensor<?xf32>, tensor<?xi64>)
// CHECK:             reducer(%[[VAL_11:.*]]: tensor<f32>, %[[VAL_13:.*]]: tensor<f32>) (%[[VAL_12:.*]]: tensor<i64>, %[[VAL_14:.*]]: tensor<i64>)  {
// CHECK:             %[[VAL_15:.*]] = mhlo.compare GE, %[[VAL_11]], %[[VAL_13]], FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             %[[VAL_16:.*]] = "mhlo.select"(%[[VAL_15]], %[[VAL_11]], %[[VAL_13]]) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:             %[[VAL_17:.*]] = mhlo.compare EQ, %[[VAL_11]], %[[VAL_13]], FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             %[[VAL_18:.*]] = mhlo.minimum %[[VAL_12]], %[[VAL_14]] : tensor<i64>
// CHECK:             %[[VAL_19:.*]] = "mhlo.select"(%[[VAL_15]], %[[VAL_12]], %[[VAL_14]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:             %[[VAL_20:.*]] = "mhlo.select"(%[[VAL_17]], %[[VAL_18]], %[[VAL_19]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:             "mhlo.return"(%[[VAL_16]], %[[VAL_20]]) : (tensor<f32>, tensor<i64>) -> ()
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_22:.*]] = tensor.from_elements %[[VAL_3]], %[[VAL_21]] : tensor<2xi64>
// CHECK:           %[[VAL_23:.*]] = "mhlo.dynamic_reshape"(%[[VAL_10]]#0, %[[VAL_22]]) : (tensor<?xf32>, tensor<2xi64>) -> tensor<?x1xf32>
// CHECK:           %[[VAL_24:.*]] = "mhlo.dynamic_reshape"(%[[VAL_10]]#1, %[[VAL_22]]) : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
// CHECK:           %[[VAL_25:.*]] = torch_c.from_builtin_tensor %[[VAL_23]] : tensor<?x1xf32> -> !torch.vtensor<[?,1],f32>
// CHECK:           %[[VAL_26:.*]] = torch_c.from_builtin_tensor %[[VAL_24]] : tensor<?x1xi64> -> !torch.vtensor<[?,1],si64>
// CHECK:           return %[[VAL_25]], %[[VAL_26]] : !torch.vtensor<[?,1],f32>, !torch.vtensor<[?,1],si64>

func.func @torch.aten.max.dim$keepdim(%arg0: !torch.vtensor<[?,?],f32>) -> (!torch.vtensor<[?,1],f32>, !torch.vtensor<[?,1],si64>) {
  %true = torch.constant.bool true
  %int1 = torch.constant.int 1
  %values, %indices = torch.aten.max.dim %arg0, %int1, %true : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?,1],f32>, !torch.vtensor<[?,1],si64>
  return %values, %indices : !torch.vtensor<[?,1],f32>, !torch.vtensor<[?,1],si64>
}

// -----
// CHECK-LABEL:   func.func @torch.aten.max.dim(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> (!torch.vtensor<[?],f32>, !torch.vtensor<[?],si64>) {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %false = torch.constant.bool false
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %[[IDX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = tensor.dim %[[VAL_1]], %[[IDX_0]] : tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_2]] : index to i64
// CHECK:           %[[IDX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = tensor.dim %[[VAL_1]], %[[IDX_1]] : tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = arith.index_cast %[[VAL_4]] : index to i64
// CHECK:           %[[VAL_6:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:           %[[VAL_7:.*]] = mhlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_8:.*]] = tensor.from_elements %[[VAL_3]], %[[VAL_5]] : tensor<2xi64>
// CHECK:           %[[VAL_9:.*]] = "mhlo.dynamic_iota"(%[[VAL_8]]) {iota_dimension = 1 : i64} : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:           %[[VAL_10:.*]]:2 = mhlo.reduce(%[[VAL_1]] init: %[[VAL_6]]), (%[[VAL_9]] init: %[[VAL_7]]) across dimensions = [1] : (tensor<?x?xf32>, tensor<?x?xi64>, tensor<f32>, tensor<i64>) -> (tensor<?xf32>, tensor<?xi64>)
// CHECK:             reducer(%[[VAL_11:.*]]: tensor<f32>, %[[VAL_13:.*]]: tensor<f32>) (%[[VAL_12:.*]]: tensor<i64>, %[[VAL_14:.*]]: tensor<i64>)  {
// CHECK:             %[[VAL_15:.*]] = mhlo.compare GE, %[[VAL_11]], %[[VAL_13]], FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             %[[VAL_16:.*]] = "mhlo.select"(%[[VAL_15]], %[[VAL_11]], %[[VAL_13]]) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:             %[[VAL_17:.*]] = mhlo.compare EQ, %[[VAL_11]], %[[VAL_13]], FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             %[[VAL_18:.*]] = mhlo.minimum %[[VAL_12]], %[[VAL_14]] : tensor<i64>
// CHECK:             %[[VAL_19:.*]] = "mhlo.select"(%[[VAL_15]], %[[VAL_12]], %[[VAL_14]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:             %[[VAL_20:.*]] = "mhlo.select"(%[[VAL_17]], %[[VAL_18]], %[[VAL_19]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:             "mhlo.return"(%[[VAL_16]], %[[VAL_20]]) : (tensor<f32>, tensor<i64>) -> ()
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = torch_c.from_builtin_tensor %[[VAL_10]]#0 : tensor<?xf32> -> !torch.vtensor<[?],f32>
// CHECK:           %[[VAL_22:.*]] = torch_c.from_builtin_tensor %[[VAL_10]]#1 : tensor<?xi64> -> !torch.vtensor<[?],si64>
// CHECK:           return %[[VAL_21]], %[[VAL_22]] : !torch.vtensor<[?],f32>, !torch.vtensor<[?],si64>
func.func @torch.aten.max.dim(%arg0: !torch.vtensor<[?,?],f32>) -> (!torch.vtensor<[?],f32>, !torch.vtensor<[?],si64>) {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %values, %indices = torch.aten.max.dim %arg0, %int1, %false : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?],f32>, !torch.vtensor<[?],si64>
  return %values, %indices : !torch.vtensor<[?],f32>, !torch.vtensor<[?],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.argmax$keepdim(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,1],si64> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %true = torch.constant.bool true
// CHECK:           %[[IDX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = tensor.dim %[[VAL_1]], %[[IDX_0]] : tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_2]] : index to i64
// CHECK:           %[[IDX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = tensor.dim %[[VAL_1]], %[[IDX_1]] : tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = arith.index_cast %[[VAL_4]] : index to i64
// CHECK:           %[[VAL_6:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:           %[[VAL_7:.*]] = mhlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_8:.*]] = tensor.from_elements %[[VAL_3]], %[[VAL_5]] : tensor<2xi64>
// CHECK:           %[[VAL_9:.*]] = "mhlo.dynamic_iota"(%[[VAL_8]]) {iota_dimension = 1 : i64} : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:           %[[VAL_10:.*]]:2 = mhlo.reduce(%[[VAL_1]] init: %[[VAL_6]]), (%[[VAL_9]] init: %[[VAL_7]]) across dimensions = [1] : (tensor<?x?xf32>, tensor<?x?xi64>, tensor<f32>, tensor<i64>) -> (tensor<?xf32>, tensor<?xi64>)
// CHECK:             reducer(%[[VAL_11:.*]]: tensor<f32>, %[[VAL_13:.*]]: tensor<f32>) (%[[VAL_12:.*]]: tensor<i64>, %[[VAL_14:.*]]: tensor<i64>)  {
// CHECK:             %[[VAL_15:.*]] = mhlo.compare GE, %[[VAL_11]], %[[VAL_13]], FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             %[[VAL_16:.*]] = "mhlo.select"(%[[VAL_15]], %[[VAL_11]], %[[VAL_13]]) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:             %[[VAL_17:.*]] = mhlo.compare EQ, %[[VAL_11]], %[[VAL_13]], FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             %[[VAL_18:.*]] = mhlo.minimum %[[VAL_12]], %[[VAL_14]] : tensor<i64>
// CHECK:             %[[VAL_19:.*]] = "mhlo.select"(%[[VAL_15]], %[[VAL_12]], %[[VAL_14]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:             %[[VAL_20:.*]] = "mhlo.select"(%[[VAL_17]], %[[VAL_18]], %[[VAL_19]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:             "mhlo.return"(%[[VAL_16]], %[[VAL_20]]) : (tensor<f32>, tensor<i64>) -> ()
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_22:.*]] = tensor.from_elements %[[VAL_3]], %[[VAL_21]] : tensor<2xi64>
// CHECK:           %[[VAL_23:.*]] = "mhlo.dynamic_reshape"(%[[VAL_10]]#1, %[[VAL_22]]) : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
// CHECK:           %[[VAL_24:.*]] = torch_c.from_builtin_tensor %[[VAL_23]] : tensor<?x1xi64> -> !torch.vtensor<[?,1],si64>
// CHECK:           return %[[VAL_24]] : !torch.vtensor<[?,1],si64>
func.func @torch.aten.argmax$keepdim(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,1],si64> {
  %int1 = torch.constant.int 1
  %true = torch.constant.bool true
  %indices = torch.aten.argmax %arg0, %int1, %true : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?,1],si64>
  return %indices : !torch.vtensor<[?,1],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.argmax(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?],si64> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %false = torch.constant.bool false
// CHECK:           %[[IDX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = tensor.dim %[[VAL_1]], %[[IDX_0]] : tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_2]] : index to i64
// CHECK:           %[[IDX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = tensor.dim %[[VAL_1]], %[[IDX_1]] : tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = arith.index_cast %[[VAL_4]] : index to i64
// CHECK:           %[[VAL_6:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:           %[[VAL_7:.*]] = mhlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_8:.*]] = tensor.from_elements %[[VAL_3]], %[[VAL_5]] : tensor<2xi64>
// CHECK:           %[[VAL_9:.*]] = "mhlo.dynamic_iota"(%[[VAL_8]]) {iota_dimension = 1 : i64} : (tensor<2xi64>) -> tensor<?x?xi64>
// CHECK:           %[[VAL_10:.*]]:2 = mhlo.reduce(%[[VAL_1]] init: %[[VAL_6]]), (%[[VAL_9]] init: %[[VAL_7]]) across dimensions = [1] : (tensor<?x?xf32>, tensor<?x?xi64>, tensor<f32>, tensor<i64>) -> (tensor<?xf32>, tensor<?xi64>)
// CHECK:             reducer(%[[VAL_11:.*]]: tensor<f32>, %[[VAL_13:.*]]: tensor<f32>) (%[[VAL_12:.*]]: tensor<i64>, %[[VAL_14:.*]]: tensor<i64>)  {
// CHECK:             %[[VAL_15:.*]] = mhlo.compare GE, %[[VAL_11]], %[[VAL_13]], FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             %[[VAL_16:.*]] = "mhlo.select"(%[[VAL_15]], %[[VAL_11]], %[[VAL_13]]) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:             %[[VAL_17:.*]] = mhlo.compare EQ, %[[VAL_11]], %[[VAL_13]], FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK:             %[[VAL_18:.*]] = mhlo.minimum %[[VAL_12]], %[[VAL_14]] : tensor<i64>
// CHECK:             %[[VAL_19:.*]] = "mhlo.select"(%[[VAL_15]], %[[VAL_12]], %[[VAL_14]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:             %[[VAL_20:.*]] = "mhlo.select"(%[[VAL_17]], %[[VAL_18]], %[[VAL_19]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:             "mhlo.return"(%[[VAL_16]], %[[VAL_20]]) : (tensor<f32>, tensor<i64>) -> ()
// CHECK:           }
// CHECK:           %[[VAL_11:.*]] = torch_c.from_builtin_tensor %[[VAL_10]]#1 : tensor<?xi64> -> !torch.vtensor<[?],si64>
// CHECK:           return %[[VAL_11]] : !torch.vtensor<[?],si64>
func.func @torch.aten.argmax(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?],si64> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %indices = torch.aten.argmax %arg0, %int1, %false : !torch.vtensor<[?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?],si64>
  return %indices : !torch.vtensor<[?],si64>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.sum.dim_Intlist$keepdim(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[1,1,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %true = torch.constant.bool true
// CHECK:           %none = torch.constant.none
// CHECK:           %[[VAL_2:.*]] = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_3:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_4:.*]] = mhlo.reduce(%[[VAL_1:.*]] init: %[[VAL_3:.*]]) applies mhlo.add across dimensions = [0, 1] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?xf32>
// CHECK:           %[[IDX_0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = tensor.dim %[[VAL_1]], %[[IDX_0]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_6:.*]] = arith.index_cast %[[VAL_5]] : index to i64
// CHECK:           %[[IDX_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = tensor.dim %[[VAL_1]], %[[IDX_1]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_8:.*]] = arith.index_cast %[[VAL_7]] : index to i64
// CHECK:           %[[IDX_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_9:.*]] = tensor.dim %[[VAL_1]], %[[IDX_2]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_10:.*]] = arith.index_cast %[[VAL_9]] : index to i64
// CHECK:           %[[ONE_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_11:.*]] = tensor.from_elements %[[ONE_0]], %[[ONE_0]], %[[VAL_10]] : tensor<3xi64>
// CHECK:           %[[VAL_12:.*]] = "mhlo.dynamic_reshape"(%[[VAL_4]], %[[VAL_11]]) : (tensor<?xf32>, tensor<3xi64>) -> tensor<1x1x?xf32>
// CHECK:           %[[VAL_13:.*]] = torch_c.from_builtin_tensor %[[VAL_12]] : tensor<1x1x?xf32> -> !torch.vtensor<[1,1,?],f32>
// CHECK:           return %[[VAL_13]] : !torch.vtensor<[1,1,?],f32>
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

// CHECK-LABEL:   func.func @torch.aten.sum.dim_Intlist(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?],f32> {
// CHECK:           %[[VAL_01:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %false = torch.constant.bool false
// CHECK:           %none = torch.constant.none
// CHECK:           %[[VAL_2:.*]] = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_3:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_4:.*]] = mhlo.reduce(%[[VAL_1]] init: %[[VAL_3]]) applies mhlo.add across dimensions = [0, 1] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?xf32>
// CHECK:           %[[VAL_5:.*]] = torch_c.from_builtin_tensor %[[VAL_4]] : tensor<?xf32> -> !torch.vtensor<[?],f32>
// CHECK:           return %[[VAL_5]] : !torch.vtensor<[?],f32>
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

// CHECK-LABEL:   func.func @torch.aten.sum(
// CHECK-SAME:                         %[[VAL_0:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:           %none = torch.constant.none
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = mhlo.reduce(%[[VAL_1]] init: %[[VAL_2]]) applies mhlo.add across dimensions = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[],f32>
func.func @torch.aten.sum(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
  %none = torch.constant.none
  %0 = torch.aten.sum %arg0, %none : !torch.vtensor<[?,?,?],f32>, !torch.none -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.max(
// CHECK-SAME:                         %[[VAL_0:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:           %[[VAL_2:.*]] = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = mhlo.reduce(%[[VAL_1]] init: %[[VAL_2]]) applies mhlo.maximum across dimensions = [0, 1, 2] : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<f32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[],f32>
func.func @torch.aten.max(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[],f32> {
  %0 = torch.aten.max %arg0 : !torch.vtensor<[?,?,?],f32> -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}
