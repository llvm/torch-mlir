// RUN: torch-mlir-opt <%s -convert-torch-to-linalg | FileCheck %s

// Test that AtenPowTensorTensorOp with integer result type properly rounds
func.func @test_aten_pow_tensor_tensor_int_result(%arg0: !torch.vtensor<[3],si32>, %arg1: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],si32> {
  %0 = torch.aten.pow.Tensor_Tensor %arg0, %arg1 : !torch.vtensor<[3],si32>, !torch.vtensor<[3],f32> -> !torch.vtensor<[3],si32>
  return %0 : !torch.vtensor<[3],si32>
}

// CHECK-LABEL: func.func @test_aten_pow_tensor_tensor_int_result
// CHECK:         %[[SITOFP:.+]] = arith.sitofp %{{.*}} : i32 to f64
// CHECK:         %[[EXTF:.+]] = arith.extf %{{.*}} : f32 to f64
// CHECK:         %[[POW:.+]] = math.powf %[[SITOFP]], %[[EXTF]] : f64
// CHECK:         %[[ROUND:.+]] = math.roundeven %[[POW]] : f64
// CHECK:         %[[FPTOSI:.+]] = arith.fptosi %[[ROUND]] : f64 to i32
// CHECK:         linalg.yield %[[FPTOSI]] : i32

// Test with float inputs but integer result
func.func @test_aten_pow_float_inputs_int_result(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],si32> {
  %0 = torch.aten.pow.Tensor_Tensor %arg0, %arg1 : !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32> -> !torch.vtensor<[3],si32>
  return %0 : !torch.vtensor<[3],si32>
}

// CHECK-LABEL: func.func @test_aten_pow_float_inputs_int_result
// CHECK:         %[[EXTF0:.+]] = arith.extf %{{.*}} : f32 to f64
// CHECK:         %[[EXTF1:.+]] = arith.extf %{{.*}} : f32 to f64
// CHECK:         %[[POW:.+]] = math.powf %[[EXTF0]], %[[EXTF1]] : f64
// CHECK:         %[[ROUND:.+]] = math.roundeven %[[POW]] : f64
// CHECK:         %[[FPTOSI:.+]] = arith.fptosi %[[ROUND]] : f64 to i32
// CHECK:         linalg.yield %[[FPTOSI]] : i32

// Test with float result (no rounding should occur)
func.func @test_aten_pow_float_result(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> {
  %0 = torch.aten.pow.Tensor_Tensor %arg0, %arg1 : !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32> -> !torch.vtensor<[3],f32>
  return %0 : !torch.vtensor<[3],f32>
}

// CHECK-LABEL: func.func @test_aten_pow_float_result
// CHECK:         %[[POW:.+]] = math.powf %{{.*}}, %{{.*}} : f32
// CHECK-NOT:     math.roundeven
// CHECK:         linalg.yield %[[POW]] : f32
