// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: #map
// CHECK-LABEL: func @grid_sampler
// CHECK:  %[[TC0:.*]] = torch_c.to_builtin_tensor %[[ARG0:.*]] : !torch.vtensor<[4,10,10,4],f32> -> tensor<4x10x10x4xf32>
// CHECK:  %[[TC1:.*]] = torch_c.to_builtin_tensor %[[ARG1:.*]] : !torch.vtensor<[4,6,8,2],f32> -> tensor<4x6x8x2xf32>
// CHECK:  %[[FALSE:.*]] = torch.constant.bool false
// CHECK:  %[[C0:.*]] = arith.constant 0 : index
// CHECK:  %[[C1:.*]] = arith.constant 1 : index
// CHECK:  %[[C2:.*]] = arith.constant 2 : index
// CHECK:  %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:  %[[CST1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:  %[[CST2:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:  %[[C2_3:.*]] = arith.constant 2 : index
// CHECK:  %[[DIM:.*]] = tensor.dim %[[TC0]], %[[C2_3]] : tensor<4x10x10x4xf32>
// CHECK:  %[[C3:.*]] = arith.constant 3 : index
// CHECK:  %[[DIM_4:.*]] = tensor.dim %[[TC0]], %[[C3]] : tensor<4x10x10x4xf32>
// CHECK:  %[[X2:.*]] = arith.subi %[[DIM:.*]], %[[C1]] : index
// CHECK:  %[[X3:.*]] = arith.subi %[[DIM_4]], %[[C1:.*]] : index
// CHECK:  %[[X4:.*]] = arith.index_cast %[[X2]] : index to i64
// CHECK:  %[[X5:.*]] = arith.index_cast %[[X3]] : index to i64
// CHECK:  %[[X6:.*]] = arith.sitofp %[[X4]] : i64 to f32
// CHECK:  %[[X7:.*]] = arith.sitofp %[[X5]] : i64 to f32
// CHECK:  %[[X8:.*]] = arith.divf %[[X6]], %[[CST2]] : f32
// CHECK:  %[[X9:.*]] = arith.divf %[[X7]], %[[CST2]] : f32
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
// CHECK:  %[[X15:.*]] = arith.mulf %[[X13:.*]], %[[X8:.*]] : f32
// CHECK:  %[[X16:.*]] = arith.mulf %[[X14:.*]], %[[X9:.*]] : f32
// CHECK:  %[[X40:.*]] = arith.mulf %[[EXTRACTED:.*]], %[[X39:.*]] : f32
// CHECK:  %[[X41:.*]] = arith.mulf %[[X31:.*]], %[[X37:.*]] : f32
// CHECK:  %[[X42:.*]] = arith.addf %[[X40:.*]], %[[X41]] : f32
// CHECK:   %[[X43:.*]] = arith.subf %[[CST_1:.*]], %[[X37]] : f32
// CHECK:   %[[X44:.*]] = arith.mulf %[[X32:.*]], %[[X43]] : f32
// CHECK:   %[[X45:.*]] = arith.mulf %[[X34:.*]], %[[X37]] : f32
// CHECK:   %[[X46:.*]] = arith.addf %[[X44]], %[[X45]] : f32
// CHECK:   %[[X47:.*]] = arith.subf %[[CST_1]], %[[X38:.*]] : f32
// CHECK:   %[[X48:.*]] = arith.mulf %[[X42]], %[[X47]] : f32
// CHECK:   %[[X49:.*]] = arith.mulf %[[X46]], %[[X38]] : f32
// CHECK:   %[[X50:.*]] = arith.addf %[[X48]], %[[X49]] : f32
// CHECK:   linalg.yield %[[X50]] : f32
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