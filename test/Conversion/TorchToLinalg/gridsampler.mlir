// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: #map
// CHECK-LABEL: func @grid_sampler
// CHECK-DAG:  %[[TC0:.*]] = torch_c.to_builtin_tensor %[[ARG0:.*]] : !torch.vtensor<[4,10,10,4],f32> -> tensor<4x10x10x4xf32>
// CHECK-DAG:  %[[TC1:.*]] = torch_c.to_builtin_tensor %[[ARG1:.*]] : !torch.vtensor<[4,6,8,2],f32> -> tensor<4x6x8x2xf32>
// CHECK-DAG:  %[[FALSE:.*]] = torch.constant.bool false
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:  %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:  %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:  %[[CST1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:  %[[CST2:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:  %[[C2_3:.*]] = arith.constant 2 : index
// CHECK-DAG:  %[[DIM:.*]] = tensor.dim %[[TC0]], %[[C2_3]] : tensor<4x10x10x4xf32>
// CHECK-DAG:  %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:  %[[DIM_4:.*]] = tensor.dim %[[TC0]], %[[C3]] : tensor<4x10x10x4xf32>
// CHECK-DAG:  %[[X2:.*]] = arith.subi %[[DIM:.*]], %[[C1]] : index
// CHECK-DAG:  %[[X3:.*]] = arith.subi %[[DIM_4]], %[[C1:.*]] : index
// CHECK-DAG:  %[[X4:.*]] = arith.index_cast %[[X2]] : index to i64
// CHECK-DAG:  %[[X5:.*]] = arith.index_cast %[[X3]] : index to i64
// CHECK-DAG:  %[[X6:.*]] = arith.sitofp %[[X4]] : i64 to f32
// CHECK-DAG:  %[[X7:.*]] = arith.sitofp %[[X5]] : i64 to f32
// CHECK-DAG:  %[[X8:.*]] = arith.divf %[[X6]], %[[CST2]] : f32
// CHECK-DAG:  %[[X9:.*]] = arith.divf %[[X7]], %[[CST2]] : f32
func.func @grid_sampler(%arg0: !torch.vtensor<[4,10,10,4],f32>, %arg1: !torch.vtensor<[4,6,8,2],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %true = torch.constant.bool 0
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 0
  %4 = torch.aten.grid_sampler %arg0, %arg1, %int0, %int1, %true : !torch.vtensor<[4,10,10,4],f32>, !torch.vtensor<[4,6,8,2],f32>, !torch.int, !torch.int, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  return %4 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL: func @grid_sampler2
// CHECK: #map
// CHECK-DAG:  %[[X15:.*]] = arith.mulf %[[X13:.*]], %[[X8:.*]] : f32
// CHECK-DAG:  %[[X16:.*]] = arith.mulf %[[X14:.*]], %[[X9:.*]] : f32
// CHECK-DAG:  %[[X40:.*]] = arith.mulf %[[EXTRACTED:.*]], %[[X39:.*]] : f32
// CHECK-DAG:  %[[X41:.*]] = arith.mulf %[[X31:.*]], %[[X37:.*]] : f32
// CHECK-DAG:  %[[X42:.*]] = arith.addf %[[X40:.*]], %[[X41]] : f32
// CHECK-DAG:   %[[X43:.*]] = arith.subf %[[CST_1:.*]], %[[X37]] : f32
// CHECK-DAG:   %[[X45:.*]] = arith.mulf %[[X34:.*]], %[[X37]] : f32
// CHECK-DAG:   %[[X46:.*]] = arith.addf %[[X44:.*]], %[[X45]] : f32
// CHECK-DAG:   %[[X47:.*]] = arith.subf %[[CST_1]], %[[X38:.*]] : f32
// CHECK-DAG:   %[[X48:.*]] = arith.mulf %[[X42]], %[[XX47:.*]] : f32
// CHECK-DAG:   %[[X49:.*]] = arith.mulf %[[X46]], %[[XX38:.*]] : f32
// CHECK-DAG:   %[[X50:.*]] = arith.addf %[[X48]], %[[X49]] : f32
// CHECK-DAG:   linalg.yield %[[X50]] : f32
// CHECK: } -> tensor<?x?x?x?xf32>
// CHECK: %[[X12:.*]] = torch_c.from_builtin_tensor %[[X11:.*]] : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK: return %[[X12]] : !torch.vtensor<[?,?,?,?],f32>
func.func @grid_sampler2(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %true = torch.constant.bool 0
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 0
  %4 = torch.aten.grid_sampler %arg0, %arg1, %int0, %int1, %true : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,?,?],f32>, !torch.int, !torch.int, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  return %4 : !torch.vtensor<[?,?,?,?],f32>
}