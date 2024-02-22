// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: #map
// CHECK-LABEL: func @grid_sampler
// CHECK:  %0 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[4,10,10,4],f32> -> tensor<4x10x10x4xf32>
// CHECK:  %1 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[4,6,8,2],f32> -> tensor<4x6x8x2xf32>
// CHECK:  %false = torch.constant.bool false
// CHECK:  %int0 = torch.constant.int 0
// CHECK:  %int0_0 = torch.constant.int 0
// CHECK:  %c0 = arith.constant 0 : index
// CHECK:  %c1 = arith.constant 1 : index
// CHECK:  %c2 = arith.constant 2 : index
// CHECK:  %cst = arith.constant 0.000000e+00 : f32
// CHECK:  %cst_1 = arith.constant 1.000000e+00 : f32
// CHECK:  %cst_2 = arith.constant 2.000000e+00 : f32
// CHECK:  %c2_3 = arith.constant 2 : index
// CHECK:  %dim = tensor.dim %0, %c2_3 : tensor<4x10x10x4xf32>
// CHECK:  %c3 = arith.constant 3 : index
// CHECK:  %dim_4 = tensor.dim %0, %c3 : tensor<4x10x10x4xf32>
// CHECK:  %2 = arith.subi %dim, %c1 : index
// CHECK:  %3 = arith.subi %dim_4, %c1 : index
// CHECK:  %4 = arith.index_cast %2 : index to i64
// CHECK:  %5 = arith.index_cast %3 : index to i64
// CHECK:  %6 = arith.sitofp %4 : i64 to f32
// CHECK:  %7 = arith.sitofp %5 : i64 to f32
// CHECK:  %8 = arith.divf %6, %cst_2 : f32
// CHECK:  %9 = arith.divf %7, %cst_2 : f32
// CHECK:  %c0_5 = arith.constant 0 : index
// CHECK:  %c4 = arith.constant 4 : index
// CHECK:  %c1_6 = arith.constant 1 : index
// CHECK:  %c6 = arith.constant 6 : index
// CHECK:  %c2_7 = arith.constant 2 : index
// CHECK:  %c8 = arith.constant 8 : index
// CHECK:  %c3_8 = arith.constant 3 : index
// CHECK:  %c2_9 = arith.constant 2 : index
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
// CHECK:   %41 = arith.mulf %31, %37 : f32
// CHECK:   %42 = arith.addf %40, %41 : f32
// CHECK:   %43 = arith.subf %cst_1, %37 : f32
// CHECK:   %44 = arith.mulf %32, %43 : f32
// CHECK:   %45 = arith.mulf %34, %37 : f32
// CHECK:   %46 = arith.addf %44, %45 : f32
// CHECK:   %47 = arith.subf %cst_1, %38 : f32
// CHECK:   %48 = arith.mulf %42, %47 : f32
// CHECK:   %49 = arith.mulf %46, %38 : f32
// CHECK:   %50 = arith.addf %48, %49 : f32
// CHECK:   linalg.yield %50 : f32
// CHECK: } -> tensor<?x?x?x?xf32>
// CHECK: %12 = torch_c.from_builtin_tensor %11 : tensor<?x?x?x?xf32> -> !torch.vtensor<[?,?,?,?],f32>
// CHECK: return %12 : !torch.vtensor<[?,?,?,?],f32>
func.func @grid_sampler2(%arg0: !torch.vtensor<[?,?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %true = torch.constant.bool 0
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 0
  %4 = torch.aten.grid_sampler %arg0, %arg1, %int0, %int1, %true : !torch.vtensor<[?,?,?,?],f32>, !torch.vtensor<[?,?,?,?],f32>, !torch.int, !torch.int, !torch.bool -> !torch.vtensor<[?,?,?,?],f32>
  return %4 : !torch.vtensor<[?,?,?,?],f32>
}