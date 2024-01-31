// RUN: torch-mlir-opt -pass-pipeline='builtin.module(onnx-backend-to-linalg-on-tensors-backend-pipeline)' -split-input-file %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (0, d1, d2)>
// CHECK-LABEL:   ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
// CHECK-LABEL:   func.func @test_softmax_axis_0(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
func.func @test_softmax_axis_0(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 0xFF800000 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<1x4x5xi64>
// CHECK:           %[[VAL_5:.*]] = linalg.fill ins(%[[VAL_1]] : i64) outs(%[[VAL_4]] : tensor<1x4x5xi64>) -> tensor<1x4x5xi64>
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<1x4x5xf32>
// CHECK:           %[[VAL_7:.*]] = linalg.fill ins(%[[VAL_2]] : f32) outs(%[[VAL_6]] : tensor<1x4x5xf32>) -> tensor<1x4x5xf32>
// CHECK:           %[[VAL_8:.*]]:2 = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["reduction", "parallel", "parallel"]} ins(%[[VAL_0]] : tensor<3x4x5xf32>) outs(%[[VAL_7]], %[[VAL_5]] : tensor<1x4x5xf32>, tensor<1x4x5xi64>) {
// CHECK:           ^bb0(%[[VAL_9:.*]]: f32, %[[VAL_10:.*]]: f32, %[[VAL_11:.*]]: i64):
// CHECK:             %[[VAL_12:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_13:.*]] = arith.index_cast %[[VAL_12]] : index to i64
// CHECK:             %[[VAL_14:.*]] = arith.maximumf %[[VAL_9]], %[[VAL_10]] : f32
// CHECK:             %[[VAL_15:.*]] = arith.cmpf ogt, %[[VAL_9]], %[[VAL_10]] : f32
// CHECK:             %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_13]], %[[VAL_11]] : i64
// CHECK:             linalg.yield %[[VAL_14]], %[[VAL_16]] : f32, i64
// CHECK:           } -> (tensor<1x4x5xf32>, tensor<1x4x5xi64>)
// CHECK:           %[[VAL_17:.*]] = tensor.empty() : tensor<3x4x5xf32>
// CHECK:           %[[VAL_18:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_0]], %[[VAL_19:.*]]#0 : tensor<3x4x5xf32>, tensor<1x4x5xf32>) outs(%[[VAL_17]] : tensor<3x4x5xf32>) {
// CHECK:           ^bb0(%[[VAL_20:.*]]: f32, %[[VAL_21:.*]]: f32, %[[VAL_22:.*]]: f32):
// CHECK:             %[[VAL_23:.*]] = arith.subf %[[VAL_20]], %[[VAL_21]] : f32
// CHECK:             linalg.yield %[[VAL_23]] : f32
// CHECK:           } -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_24:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_18]] : tensor<3x4x5xf32>) outs(%[[VAL_17]] : tensor<3x4x5xf32>) {
// CHECK:           ^bb0(%[[VAL_25:.*]]: f32, %[[VAL_26:.*]]: f32):
// CHECK:             %[[VAL_27:.*]] = math.exp %[[VAL_25]] : f32
// CHECK:             linalg.yield %[[VAL_27]] : f32
// CHECK:           } -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_28:.*]] = linalg.fill ins(%[[VAL_3]] : f32) outs(%[[VAL_6]] : tensor<1x4x5xf32>) -> tensor<1x4x5xf32>
// CHECK:           %[[VAL_29:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["reduction", "parallel", "parallel"]} ins(%[[VAL_24]] : tensor<3x4x5xf32>) outs(%[[VAL_28]] : tensor<1x4x5xf32>) {
// CHECK:           ^bb0(%[[VAL_30:.*]]: f32, %[[VAL_31:.*]]: f32):
// CHECK:             %[[VAL_32:.*]] = arith.addf %[[VAL_30]], %[[VAL_31]] : f32
// CHECK:             linalg.yield %[[VAL_32]] : f32
// CHECK:           } -> tensor<1x4x5xf32>
// CHECK:           %[[VAL_33:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_24]], %[[VAL_29]] : tensor<3x4x5xf32>, tensor<1x4x5xf32>) outs(%[[VAL_17]] : tensor<3x4x5xf32>) {
// CHECK:           ^bb0(%[[VAL_34:.*]]: f32, %[[VAL_35:.*]]: f32, %[[VAL_36:.*]]: f32):
// CHECK:             %[[VAL_37:.*]] = arith.divf %[[VAL_34]], %[[VAL_35]] : f32
// CHECK:             linalg.yield %[[VAL_37]] : f32
// CHECK:           } -> tensor<3x4x5xf32>
// CHECK:           return %[[VAL_33]] : tensor<3x4x5xf32>
// CHECK:         }
  %0 = torch.operator "onnx.Softmax"(%arg0) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}

// -----

// CHECK: #[[$ATTR_0:.+]] = affine_map<() -> ()>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> ()>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL:   ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
// CHECK-LABEL:   func.func @test_leaky_relu(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> attributes {torch.onnx_meta.opset_version = 16 : si64} {
func.func @test_leaky_relu(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.opset_version = 16 : si64} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<i64>
// CHECK:           %[[VAL_4:.*]] = linalg.fill ins(%[[VAL_1]] : i64) outs(%[[VAL_3]] : tensor<i64>) -> tensor<i64>
// CHECK:           %[[VAL_5:.*]] = tensor.empty() : tensor<f32>
// CHECK:           %[[VAL_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = []} ins(%[[VAL_4]] : tensor<i64>) outs(%[[VAL_5]] : tensor<f32>) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i64, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.sitofp %[[VAL_7]] : i64 to f32
// CHECK:             linalg.yield %[[VAL_9]] : f32
// CHECK:           } -> tensor<f32>
// CHECK:           %[[VAL_10:.*]] = tensor.empty() : tensor<3x4x5xf32>
// CHECK:           %[[VAL_11:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_6]], %[[VAL_0]] : tensor<f32>, tensor<3x4x5xf32>) outs(%[[VAL_10]] : tensor<3x4x5xf32>) {
// CHECK:           ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32, %[[VAL_14:.*]]: f32):
// CHECK:             %[[VAL_15:.*]] = arith.cmpf ogt, %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             linalg.yield %[[VAL_16]] : f32
// CHECK:           } -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_17:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_6]], %[[VAL_0]] : tensor<f32>, tensor<3x4x5xf32>) outs(%[[VAL_10]] : tensor<3x4x5xf32>) {
// CHECK:           ^bb0(%[[VAL_18:.*]]: f32, %[[VAL_19:.*]]: f32, %[[VAL_20:.*]]: f32):
// CHECK:             %[[VAL_21:.*]] = arith.cmpf olt, %[[VAL_18]], %[[VAL_19]] : f32
// CHECK:             %[[VAL_22:.*]] = arith.select %[[VAL_21]], %[[VAL_18]], %[[VAL_19]] : f32
// CHECK:             linalg.yield %[[VAL_22]] : f32
// CHECK:           } -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_23:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_17]] : tensor<3x4x5xf32>) outs(%[[VAL_10]] : tensor<3x4x5xf32>) {
// CHECK:           ^bb0(%[[VAL_24:.*]]: f32, %[[VAL_25:.*]]: f32):
// CHECK:             %[[VAL_26:.*]] = arith.mulf %[[VAL_24]], %[[VAL_2]] : f32
// CHECK:             linalg.yield %[[VAL_26]] : f32
// CHECK:           } -> tensor<3x4x5xf32>
// CHECK:           %[[VAL_27:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_11]], %[[VAL_23]] : tensor<3x4x5xf32>, tensor<3x4x5xf32>) outs(%[[VAL_10]] : tensor<3x4x5xf32>) {
// CHECK:           ^bb0(%[[VAL_28:.*]]: f32, %[[VAL_29:.*]]: f32, %[[VAL_30:.*]]: f32):
// CHECK:             %[[VAL_31:.*]] = arith.addf %[[VAL_28]], %[[VAL_29]] : f32
// CHECK:             linalg.yield %[[VAL_31]] : f32
// CHECK:           } -> tensor<3x4x5xf32>
// CHECK:           return %[[VAL_27]] : tensor<3x4x5xf32>
// CHECK:         }
  %0 = torch.operator "onnx.LeakyRelu"(%arg0) {torch.onnx.alpha = 2.000000e+00 : f32} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
  return %0 : !torch.vtensor<[3,4,5],f32>
}
