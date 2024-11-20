// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -canonicalize -split-input-file -verify-diagnostics | FileCheck %s

// CHECK:         #map = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK:         #map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK:         #map2 = affine_map<(d0, d1, d2) -> (d0, d2)>

// CHECK-LABEL:   func.func @torch.aten.fft_rfft$2d_last_dim(
// CHECK-SAME:           %arg0: !torch.vtensor<[16,9],f32>) -> !torch.vtensor<[16,5],complex<f32>> {
// CHECK-DAG:         %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:         %[[CST_0:.*]] = arith.constant dense<{{.*}}> : tensor<9x5xcomplex<f32>>
// CHECK:             %[[VAR0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[16,9],f32> -> tensor<16x9xf32>
// CHECK-DAG:         %[[VAR1:.*]] = tensor.empty() : tensor<16x5xcomplex<f32>>
// CHECK:             %[[VAR2:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[VAR1]] : tensor<16x5xcomplex<f32>>) -> tensor<16x5xcomplex<f32>>
// CHECK:             %[[VAR3:.*]] = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[VAR0]], %[[CST_0]] : tensor<16x9xf32>, tensor<9x5xcomplex<f32>>) outs(%[[VAR2]] : tensor<16x5xcomplex<f32>>) {
// CHECK:             ^bb0(%in: f32, %in_1: complex<f32>, %out: complex<f32>):
// CHECK:               %[[VAR5:.*]] = complex.re %in_1 : complex<f32>
// CHECK:               %[[VAR6:.*]] = complex.im %in_1 : complex<f32>
// CHECK:               %[[VAR7:.*]] = arith.mulf %in, %[[VAR5]] : f32
// CHECK:               %[[VAR8:.*]] = arith.mulf %in, %[[VAR6]] : f32
// CHECK:               %[[VAR9:.*]] = complex.create %[[VAR7]], %[[VAR8]] : complex<f32>
// CHECK:               %[[VAR10:.*]] = complex.add %[[VAR9]], %out : complex<f32>
// CHECK:               linalg.yield %[[VAR10]] : complex<f32>
// CHECK:             } -> tensor<16x5xcomplex<f32>>
// CHECK:             %[[VAR4:.*]] = torch_c.from_builtin_tensor %[[VAR3]] : tensor<16x5xcomplex<f32>> -> !torch.vtensor<[16,5],complex<f32>>
// CHECK:             return %[[VAR4]] : !torch.vtensor<[16,5],complex<f32>>

func.func @torch.aten.fft_rfft$2d_last_dim(%arg0: !torch.vtensor<[16,9],f32>) -> !torch.vtensor<[16,5],complex<f32>> {
  %int-1 = torch.constant.int -1
  %none = torch.constant.none
  %out = torch.aten.fft_rfft %arg0, %none, %int-1, %none : !torch.vtensor<[16,9],f32>, !torch.none, !torch.int, !torch.none -> !torch.vtensor<[16,5],complex<f32>>
  return %out : !torch.vtensor<[16,5],complex<f32>>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.fft_rfft$2d_first_dim(
// CHECK-SAME:           %arg0: !torch.vtensor<[36,23],f32>) -> !torch.vtensor<[19,23],complex<f32>> {
// CHECK-DAG:         %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:         %[[CST_0:.*]] = arith.constant dense<{{.*}}> : tensor<36x19xcomplex<f32>>
// CHECK:             %[[VAR0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[36,23],f32> -> tensor<36x23xf32>
// CHECK-DAG:         %[[VAR1:.*]] = tensor.empty() : tensor<23x36xf32>
// CHECK:             %[[TRANSPOSED:.*]] = linalg.transpose ins(%[[VAR0]] : tensor<36x23xf32>) outs(%[[VAR1]] : tensor<23x36xf32>) permutation = [1, 0]
// CHECK-DAG:         %[[VAR2:.*]] = tensor.empty() : tensor<23x19xcomplex<f32>>
// CHECK:             %[[VAR3:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[VAR2]] : tensor<23x19xcomplex<f32>>) -> tensor<23x19xcomplex<f32>>
// CHECK:             %[[VAR4:.*]] = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[TRANSPOSED]], %[[CST_0]] : tensor<23x36xf32>, tensor<36x19xcomplex<f32>>) outs(%[[VAR3]] : tensor<23x19xcomplex<f32>>) {
// CHECK:             ^bb0(%in: f32, %in_2: complex<f32>, %out: complex<f32>):
// CHECK:               %[[VAR7:.*]] = complex.re %in_2 : complex<f32>
// CHECK:               %[[VAR8:.*]] = complex.im %in_2 : complex<f32>
// CHECK:               %[[VAR9:.*]] = arith.mulf %in, %[[VAR7]] : f32
// CHECK:               %[[VAR10:.*]] = arith.mulf %in, %[[VAR8]] : f32
// CHECK:               %[[VAR11:.*]] = complex.create %[[VAR9]], %[[VAR10]] : complex<f32>
// CHECK:               %[[VAR12:.*]] = complex.add %[[VAR11]], %out : complex<f32>
// CHECK:               linalg.yield %[[VAR12]] : complex<f32>
// CHECK:             } -> tensor<23x19xcomplex<f32>>
// CHECK-DAG:         %[[VAR5:.*]] = tensor.empty() : tensor<19x23xcomplex<f32>>
// CHECK:             %[[TRANSPOSED_1:.*]] = linalg.transpose ins(%[[VAR4]] : tensor<23x19xcomplex<f32>>) outs(%[[VAR5]] : tensor<19x23xcomplex<f32>>) permutation = [1, 0]
// CHECK:             %[[VAR6:.*]] = torch_c.from_builtin_tensor %[[TRANSPOSED_1]] : tensor<19x23xcomplex<f32>> -> !torch.vtensor<[19,23],complex<f32>>
// CHECK:             return %[[VAR6]] : !torch.vtensor<[19,23],complex<f32>>
func.func @torch.aten.fft_rfft$2d_first_dim(%arg0: !torch.vtensor<[36,23],f32>) -> !torch.vtensor<[19,23],complex<f32>> {
  %int0 = torch.constant.int 0
  %none = torch.constant.none
  %out = torch.aten.fft_rfft %arg0, %none, %int0, %none : !torch.vtensor<[36,23],f32>, !torch.none, !torch.int, !torch.none -> !torch.vtensor<[19,23],complex<f32>>
  return %out : !torch.vtensor<[19,23],complex<f32>>
}
