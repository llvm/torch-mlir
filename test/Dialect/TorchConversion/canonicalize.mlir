// RUN: torch-mlir-opt %s -canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @torch_c.from_i1() -> !torch.bool {
// CHECK:     %[[TRUE:.*]] = torch.constant.bool true
// CHECK:     return %[[TRUE]] : !torch.bool
func.func @torch_c.from_i1() -> !torch.bool {
  %c1_i1 = arith.constant true
  %0 = torch_c.from_i1 %c1_i1
  return %0 : !torch.bool
}

// CHECK-LABEL:   func.func @torch_c.to_i1() -> i1 {
// CHECK:     %[[C1_I1:.*]] = arith.constant true
// CHECK:     return %[[C1_I1]] : i1
func.func @torch_c.to_i1() -> i1 {
  %bool1 = torch.constant.bool true
  %0 = torch_c.to_i1 %bool1
  return %0 : i1
}

// CHECK-LABEL:   func.func @torch_c.from_i64() -> !torch.int {
// CHECK:     %[[INT5:.*]] = torch.constant.int 5
// CHECK:     return %[[INT5]] : !torch.int
func.func @torch_c.from_i64() -> !torch.int {
  %c5_i64 = arith.constant 5 : i64
  %0 = torch_c.from_i64 %c5_i64
  return %0 : !torch.int
}

// CHECK-LABEL:   func.func @torch_c.to_i64() -> i64 {
// CHECK:     %[[C5_I64:.*]] = arith.constant 5 : i64
// CHECK:     return %[[C5_I64]] : i64
func.func @torch_c.to_i64() -> i64 {
  %int5 = torch.constant.int 5
  %0 = torch_c.to_i64 %int5
  return %0 : i64
}

// CHECK-LABEL:   func.func @torch_c.from_i64$to_i64() -> i64 {
// CHECK:     %[[C5_I64:.*]] = arith.constant 5 : i64
// CHECK:     return %[[C5_I64]] : i64
func.func @torch_c.from_i64$to_i64() -> i64 {
  %c5_i64 = arith.constant 5 : i64
  %0 = torch_c.from_i64 %c5_i64
  %1 = torch_c.to_i64 %0
  return %1 : i64
}

// CHECK-LABEL:   func.func @torch_c.to_i64$from_i64() -> !torch.int {
// CHECK:     %[[INT5:.*]] = torch.constant.int 5
// CHECK:     return %[[INT5]] : !torch.int
func.func @torch_c.to_i64$from_i64() -> !torch.int {
  %int5 = torch.constant.int 5
  %0 = torch_c.to_i64 %int5
  %1 = torch_c.from_i64 %0
  return %1 : !torch.int
}

// CHECK-LABEL:   func.func @torch_c.from_f64() -> !torch.float {
// CHECK:     %[[FLOAT5:.*]] = torch.constant.float 5.000000e+00
// CHECK:     return %[[FLOAT5]] : !torch.float
func.func @torch_c.from_f64() -> !torch.float {
  %c5_f64 = arith.constant 5.000000e+00 : f64
  %0 = torch_c.from_f64 %c5_f64
  return %0 : !torch.float
}

// CHECK-LABEL:   func.func @torch_c.to_f64() -> f64 {
// CHECK:     %[[C5_f64:.*]] = arith.constant 5.000000e+00 : f64
// CHECK:     return %[[C5_f64]] : f64
func.func @torch_c.to_f64() -> f64 {
  %float5 = torch.constant.float 5.000000e+00
  %0 = torch_c.to_f64 %float5
  return %0 : f64
}

// CHECK-LABEL:   func.func @torch_c.from_f64$to_f64() -> f64 {
// CHECK:     %[[C5_f64:.*]] = arith.constant 5.000000e+00 : f64
// CHECK:     return %[[C5_f64]] : f64
func.func @torch_c.from_f64$to_f64() -> f64 {
  %c5_f64 = arith.constant 5.000000e+00 : f64
  %0 = torch_c.from_f64 %c5_f64
  %1 = torch_c.to_f64 %0
  return %1 : f64
}

// CHECK-LABEL:   func.func @torch_c.to_f64$from_f64() -> !torch.float {
// CHECK:     %[[FLOAT5:.*]] = torch.constant.float 5.000000e+00
// CHECK:     return %[[FLOAT5]] : !torch.float
func.func @torch_c.to_f64$from_f64() -> !torch.float {
  %float5 = torch.constant.float 5.000000e+00
  %0 = torch_c.to_f64 %float5
  %1 = torch_c.from_f64 %0
  return %1 : !torch.float
}

// -----

// CHECK-LABEL: func.func @torch_c.to_builtin_tensor$fold_vtensor_literal_f32() -> tensor<3xf32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>
// CHECK:         return %[[CST]] : tensor<3xf32>
func.func @torch_c.to_builtin_tensor$fold_vtensor_literal_f32() -> tensor<3xf32> {
  %0 = torch.vtensor.literal(dense<[1.0, 2.0, 3.0]> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %1 = torch_c.to_builtin_tensor %0 : !torch.vtensor<[3],f32> -> tensor<3xf32>
  return %1 : tensor<3xf32>
}

// -----

// CHECK-LABEL: func.func @torch_c.from_builtin_tensor$fold_arith_constant_f32() -> !torch.vtensor<[2,2],f32> {
// CHECK:         %[[LIT:.*]] = torch.vtensor.literal(dense<{{.*}}> : tensor<2x2xf32>) : !torch.vtensor<[2,2],f32>
// CHECK:         return %[[LIT]] : !torch.vtensor<[2,2],f32>
func.func @torch_c.from_builtin_tensor$fold_arith_constant_f32() -> !torch.vtensor<[2,2],f32> {
  %cst = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %0 = torch_c.from_builtin_tensor %cst : tensor<2x2xf32> -> !torch.vtensor<[2,2],f32>
  return %0 : !torch.vtensor<[2,2],f32>
}

// -----

// CHECK-LABEL: func.func @torch_c.to_builtin_tensor$nofold_vtensor_literal_si64
// CHECK:         torch.vtensor.literal
// CHECK:         torch_c.to_builtin_tensor
func.func @torch_c.to_builtin_tensor$nofold_vtensor_literal_si64() -> tensor<4xi64> {
  %0 = torch.vtensor.literal(dense<[1, 2, 3, 4]> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %1 = torch_c.to_builtin_tensor %0 : !torch.vtensor<[4],si64> -> tensor<4xi64>
  return %1 : tensor<4xi64>
}

// -----

// CHECK-LABEL: func.func @torch_c.from_builtin_tensor$fold_arith_constant_int() -> !torch.vtensor<[3],si64> {
// CHECK:         %[[LIT:.*]] = torch.vtensor.literal(dense<[10, 20, 30]> : tensor<3xi64>) : !torch.vtensor<[3],si64>
// CHECK:         return %[[LIT]] : !torch.vtensor<[3],si64>
func.func @torch_c.from_builtin_tensor$fold_arith_constant_int() -> !torch.vtensor<[3],si64> {
  %cst = arith.constant dense<[10, 20, 30]> : tensor<3xi64>
  %0 = torch_c.from_builtin_tensor %cst : tensor<3xi64> -> !torch.vtensor<[3],si64>
  return %0 : !torch.vtensor<[3],si64>
}

// -----

// CHECK-LABEL: func.func @torch_c.from_builtin_tensor$to_builtin_tensor$roundtrip() -> !torch.vtensor<[2],f32> {
// CHECK:         %[[LIT:.*]] = torch.vtensor.literal(dense<[5.000000e+00, 6.000000e+00]> : tensor<2xf32>) : !torch.vtensor<[2],f32>
// CHECK:         return %[[LIT]] : !torch.vtensor<[2],f32>
func.func @torch_c.from_builtin_tensor$to_builtin_tensor$roundtrip() -> !torch.vtensor<[2],f32> {
  %0 = torch.vtensor.literal(dense<[5.0, 6.0]> : tensor<2xf32>) : !torch.vtensor<[2],f32>
  %1 = torch_c.to_builtin_tensor %0 : !torch.vtensor<[2],f32> -> tensor<2xf32>
  %2 = torch_c.from_builtin_tensor %1 : tensor<2xf32> -> !torch.vtensor<[2],f32>
  return %2 : !torch.vtensor<[2],f32>
}

// -----

// CHECK-LABEL: func.func @torch_c.to_builtin_tensor$from_builtin_tensor$roundtrip() -> tensor<2xf32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<[7.000000e+00, 8.000000e+00]> : tensor<2xf32>
// CHECK:         return %[[CST]] : tensor<2xf32>
func.func @torch_c.to_builtin_tensor$from_builtin_tensor$roundtrip() -> tensor<2xf32> {
  %cst = arith.constant dense<[7.0, 8.0]> : tensor<2xf32>
  %0 = torch_c.from_builtin_tensor %cst : tensor<2xf32> -> !torch.vtensor<[2],f32>
  %1 = torch_c.to_builtin_tensor %0 : !torch.vtensor<[2],f32> -> tensor<2xf32>
  return %1 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func.func @torch_c.to_builtin_tensor$fold_scalar() -> tensor<f32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<4.200000e+01> : tensor<f32>
// CHECK:         return %[[CST]] : tensor<f32>
func.func @torch_c.to_builtin_tensor$fold_scalar() -> tensor<f32> {
  %0 = torch.vtensor.literal(dense<42.0> : tensor<f32>) : !torch.vtensor<[],f32>
  %1 = torch_c.to_builtin_tensor %0 : !torch.vtensor<[],f32> -> tensor<f32>
  return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: func.func @torch_c.from_builtin_tensor$fold_scalar() -> !torch.vtensor<[],f32> {
// CHECK:         %[[LIT:.*]] = torch.vtensor.literal(dense<4.200000e+01> : tensor<f32>) : !torch.vtensor<[],f32>
// CHECK:         return %[[LIT]] : !torch.vtensor<[],f32>
func.func @torch_c.from_builtin_tensor$fold_scalar() -> !torch.vtensor<[],f32> {
  %cst = arith.constant dense<42.0> : tensor<f32>
  %0 = torch_c.from_builtin_tensor %cst : tensor<f32> -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL: func.func @torch_c.to_builtin_tensor$fold_vtensor_literal_f64() -> tensor<2xf64> {
// CHECK:         %[[CST:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
// CHECK:         return %[[CST]] : tensor<2xf64>
func.func @torch_c.to_builtin_tensor$fold_vtensor_literal_f64() -> tensor<2xf64> {
  %0 = torch.vtensor.literal(dense<[1.0, 2.0]> : tensor<2xf64>) : !torch.vtensor<[2],f64>
  %1 = torch_c.to_builtin_tensor %0 : !torch.vtensor<[2],f64> -> tensor<2xf64>
  return %1 : tensor<2xf64>
}

// -----

// CHECK-LABEL: func.func @torch_c.from_builtin_tensor$fold_arith_constant_f64() -> !torch.vtensor<[2],f64> {
// CHECK:         %[[LIT:.*]] = torch.vtensor.literal(dense<[3.000000e+00, 4.000000e+00]> : tensor<2xf64>) : !torch.vtensor<[2],f64>
// CHECK:         return %[[LIT]] : !torch.vtensor<[2],f64>
func.func @torch_c.from_builtin_tensor$fold_arith_constant_f64() -> !torch.vtensor<[2],f64> {
  %cst = arith.constant dense<[3.0, 4.0]> : tensor<2xf64>
  %0 = torch_c.from_builtin_tensor %cst : tensor<2xf64> -> !torch.vtensor<[2],f64>
  return %0 : !torch.vtensor<[2],f64>
}

// -----

// CHECK-LABEL: func.func @torch_c.to_builtin_tensor$nofold_vtensor_literal_si32
// CHECK:         torch.vtensor.literal
// CHECK:         torch_c.to_builtin_tensor
func.func @torch_c.to_builtin_tensor$nofold_vtensor_literal_si32() -> tensor<3xi32> {
  %0 = torch.vtensor.literal(dense<[10, 20, 30]> : tensor<3xsi32>) : !torch.vtensor<[3],si32>
  %1 = torch_c.to_builtin_tensor %0 : !torch.vtensor<[3],si32> -> tensor<3xi32>
  return %1 : tensor<3xi32>
}

// -----

// CHECK-LABEL: func.func @torch_c.from_builtin_tensor$fold_arith_constant_i32() -> !torch.vtensor<[3],si32> {
// CHECK:         %[[LIT:.*]] = torch.vtensor.literal(dense<[100, 200, 300]> : tensor<3xi32>) : !torch.vtensor<[3],si32>
// CHECK:         return %[[LIT]] : !torch.vtensor<[3],si32>
func.func @torch_c.from_builtin_tensor$fold_arith_constant_i32() -> !torch.vtensor<[3],si32> {
  %cst = arith.constant dense<[100, 200, 300]> : tensor<3xi32>
  %0 = torch_c.from_builtin_tensor %cst : tensor<3xi32> -> !torch.vtensor<[3],si32>
  return %0 : !torch.vtensor<[3],si32>
}

// -----

// CHECK-LABEL: func.func @torch_c.to_builtin_tensor$fold_vtensor_literal_bool() -> tensor<3xi1> {
// CHECK:         %[[CST:.*]] = arith.constant dense<[true, false, true]> : tensor<3xi1>
// CHECK:         return %[[CST]] : tensor<3xi1>
func.func @torch_c.to_builtin_tensor$fold_vtensor_literal_bool() -> tensor<3xi1> {
  %0 = torch.vtensor.literal(dense<[true, false, true]> : tensor<3xi1>) : !torch.vtensor<[3],i1>
  %1 = torch_c.to_builtin_tensor %0 : !torch.vtensor<[3],i1> -> tensor<3xi1>
  return %1 : tensor<3xi1>
}

// -----

// CHECK-LABEL: func.func @torch_c.from_builtin_tensor$fold_arith_constant_bool() -> !torch.vtensor<[2],i1> {
// CHECK:         %[[LIT:.*]] = torch.vtensor.literal(dense<[true, false]> : tensor<2xi1>) : !torch.vtensor<[2],i1>
// CHECK:         return %[[LIT]] : !torch.vtensor<[2],i1>
func.func @torch_c.from_builtin_tensor$fold_arith_constant_bool() -> !torch.vtensor<[2],i1> {
  %cst = arith.constant dense<[true, false]> : tensor<2xi1>
  %0 = torch_c.from_builtin_tensor %cst : tensor<2xi1> -> !torch.vtensor<[2],i1>
  return %0 : !torch.vtensor<[2],i1>
}

// -----

// CHECK-LABEL: func.func @torch_c.to_builtin_tensor$fold_splat() -> tensor<4x4xf32> {
// CHECK:         %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<4x4xf32>
// CHECK:         return %[[CST]] : tensor<4x4xf32>
func.func @torch_c.to_builtin_tensor$fold_splat() -> tensor<4x4xf32> {
  %0 = torch.vtensor.literal(dense<0.0> : tensor<4x4xf32>) : !torch.vtensor<[4,4],f32>
  %1 = torch_c.to_builtin_tensor %0 : !torch.vtensor<[4,4],f32> -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @torch_c.from_builtin_tensor$fold_arith_constant_2d_int() -> !torch.vtensor<[2,3],si64> {
// CHECK:         %[[LIT:.*]] = torch.vtensor.literal(dense<{{.*}}> : tensor<2x3xi64>) : !torch.vtensor<[2,3],si64>
// CHECK:         return %[[LIT]] : !torch.vtensor<[2,3],si64>
func.func @torch_c.from_builtin_tensor$fold_arith_constant_2d_int() -> !torch.vtensor<[2,3],si64> {
  %cst = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>
  %0 = torch_c.from_builtin_tensor %cst : tensor<2x3xi64> -> !torch.vtensor<[2,3],si64>
  return %0 : !torch.vtensor<[2,3],si64>
}

// -----

// CHECK-LABEL: func.func @torch_c.to_builtin_tensor$from_builtin_tensor$roundtrip_int() -> tensor<2xi64> {
// CHECK:         %[[CST:.*]] = arith.constant dense<[99, 100]> : tensor<2xi64>
// CHECK:         return %[[CST]] : tensor<2xi64>
func.func @torch_c.to_builtin_tensor$from_builtin_tensor$roundtrip_int() -> tensor<2xi64> {
  %cst = arith.constant dense<[99, 100]> : tensor<2xi64>
  %0 = torch_c.from_builtin_tensor %cst : tensor<2xi64> -> !torch.vtensor<[2],si64>
  %1 = torch_c.to_builtin_tensor %0 : !torch.vtensor<[2],si64> -> tensor<2xi64>
  return %1 : tensor<2xi64>
}
