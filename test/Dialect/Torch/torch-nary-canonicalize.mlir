// RUN: torch-mlir-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @fold_aten_add_splat_int
func.func @fold_aten_add_splat_int() -> !torch.vtensor<[4],si64> {
  // CHECK: torch.vtensor.literal(dense<29> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %cst_7 = torch.vtensor.literal(dense<7> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %cst_11 = torch.vtensor.literal(dense<11> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %int2 = torch.constant.int 2
  %0 = torch.aten.add.Tensor %cst_7, %cst_11, %int2 : !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
  return %0 : !torch.vtensor<[4],si64>
}

// -----

// CHECK-LABEL: @fold_aten_add_splat_int_mismatch
func.func @fold_aten_add_splat_int_mismatch() -> !torch.vtensor<[4],si64> {
  // CHECK: torch.vtensor.literal(dense<29> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %cst_7 = torch.vtensor.literal(dense<7> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %cst_11 = torch.vtensor.literal(dense<11> : tensor<4xsi32>) : !torch.vtensor<[4],si32>
  %int2 = torch.constant.int 2
  %0 = torch.aten.add.Tensor %cst_7, %cst_11, %int2 : !torch.vtensor<[4],si64>, !torch.vtensor<[4],si32>, !torch.int -> !torch.vtensor<[4],si64>
  return %0 : !torch.vtensor<[4],si64>
}

// -----

// CHECK-LABEL: @fold_aten_add_splat_float
func.func @fold_aten_add_splat_float() -> !torch.vtensor<[4],f32> {
  // CHECK: torch.vtensor.literal(dense<2.900000e+01> : tensor<4xf32>)
  %int2 = torch.constant.float 2.0
  %cst_7 = torch.vtensor.literal(dense<7.0> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %cst_11 = torch.vtensor.literal(dense<11.0> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %0 = torch.aten.add.Tensor %cst_7, %cst_11, %int2 : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.float -> !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// -----

// CHECK-LABEL: @fold_aten_add_splat_float_mismatch
func.func @fold_aten_add_splat_float_mismatch() -> !torch.vtensor<[4],f32> {
  // CHECK: torch.vtensor.literal(dense<2.900000e+01> : tensor<4xf32>)
  %int2 = torch.constant.float 2.0
  %cst_7 = torch.vtensor.literal(dense<7.0> : tensor<4xf64>) : !torch.vtensor<[4],f64>
  %cst_11 = torch.vtensor.literal(dense<11.0> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %0 = torch.aten.add.Tensor %cst_7, %cst_11, %int2 : !torch.vtensor<[4],f64>, !torch.vtensor<[4],f32>, !torch.float -> !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// -----

// CHECK-LABEL: @fold_aten_add_arr0_int
func.func @fold_aten_add_arr0_int() -> !torch.vtensor<[4],si64> {
  // CHECK: torch.vtensor.literal(dense<[28, 29, 30, 31]> : tensor<4xsi64>)
  %cst_7 = torch.vtensor.literal(dense<[6,7,8,9]> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %cst_11 = torch.vtensor.literal(dense<11> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %int2 = torch.constant.int 2
  %0 = torch.aten.add.Tensor %cst_7, %cst_11, %int2 : !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
  return %0 : !torch.vtensor<[4],si64>
}


// -----

// CHECK-LABEL: @fold_aten_add_arr1_int
func.func @fold_aten_add_arr1_int() -> !torch.vtensor<[4],si64> {
  // CHECK: torch.vtensor.literal(dense<[27, 29, 31, 33]> : tensor<4xsi64>)
  %int2 = torch.constant.int 2
  %cst_7 = torch.vtensor.literal(dense<7> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %cst_11 = torch.vtensor.literal(dense<[10,11,12,13]> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %0 = torch.aten.add.Tensor %cst_7, %cst_11, %int2 : !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
  return %0 : !torch.vtensor<[4],si64>
}


// -----

// CHECK-LABEL: @fold_aten_add_arr0_float
func.func @fold_aten_add_arr0_float() -> !torch.vtensor<[4],f32> {
  // CHECK: torch.vtensor.literal(dense<[2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01]> : tensor<4xf32>)
  %int2 = torch.constant.float 2.0
  %cst_7 = torch.vtensor.literal(dense<[6.0, 7.0, 8.0, 9.0]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %cst_11 = torch.vtensor.literal(dense<11.0> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %0 = torch.aten.add.Tensor %cst_7, %cst_11, %int2 : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.float -> !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// -----

// CHECK-LABEL: @fold_aten_add_arr1_float
func.func @fold_aten_add_arr1_float() -> !torch.vtensor<[4],f32> {
  // CHECK: torch.vtensor.literal(dense<[2.700000e+01, 2.900000e+01, 3.100000e+01, 3.300000e+01]> : tensor<4xf32>)
  %fp_2 = torch.constant.float 2.0
  %cst_7 = torch.vtensor.literal(dense<7.0> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %cst_11 = torch.vtensor.literal(dense<[10.0,11.0,12.0,13.0]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %0 = torch.aten.add.Tensor %cst_7, %cst_11, %fp_2 : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.float -> !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// -----

// CHECK-LABEL: @fold_aten_sub_splat_int
func.func @fold_aten_sub_splat_int() -> !torch.vtensor<[4],si64> {
  // CHECK: torch.vtensor.literal(dense<-15> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %int_2 = torch.constant.int 2
  %cst_7 = torch.vtensor.literal(dense<7> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %cst_11 = torch.vtensor.literal(dense<11> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %0 = torch.aten.sub.Tensor %cst_7, %cst_11, %int_2 : !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
  return %0 : !torch.vtensor<[4],si64>
}

// -----

// CHECK-LABEL: @fold_aten_sub_splat_float
func.func @fold_aten_sub_splat_float() -> !torch.vtensor<[4],f32> {
  // CHECK: torch.vtensor.literal(dense<-1.500000e+01> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %fp_2 = torch.constant.float 2.0
  %cst_7 = torch.vtensor.literal(dense<7.0> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %cst_11 = torch.vtensor.literal(dense<11.0> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %0 = torch.aten.sub.Tensor %cst_7, %cst_11, %fp_2 : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>, !torch.float -> !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// -----

// CHECK-LABEL: @fold_aten_mul_splat_int
func.func @fold_aten_mul_splat_int() -> !torch.vtensor<[4],si64> {
  // CHECK: torch.vtensor.literal(dense<77> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %cst_7 = torch.vtensor.literal(dense<7> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %cst_11 = torch.vtensor.literal(dense<11> : tensor<4xsi64>) : !torch.vtensor<[4],si64>
  %0 = torch.aten.mul.Tensor %cst_7, %cst_11: !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64> -> !torch.vtensor<[4],si64>
  return %0 : !torch.vtensor<[4],si64>
}

// -----

// CHECK-LABEL: @fold_aten_mul_splat_float
func.func @fold_aten_mul_splat_float() -> !torch.vtensor<[4],f32> {
  // CHECK: torch.vtensor.literal(dense<7.700000e+01> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %cst_7 = torch.vtensor.literal(dense<7.0> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %cst_11 = torch.vtensor.literal(dense<11.0> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %0 = torch.aten.mul.Tensor %cst_7, %cst_11 : !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}
