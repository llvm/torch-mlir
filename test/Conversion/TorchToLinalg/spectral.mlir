// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -canonicalize -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.fft_rfft$2d_last_dim(
// CHECK-SAME:           %[[INPUT_VTENSOR:.*]]: !torch.vtensor<[16,9],f32>) -> !torch.vtensor<[16,5],complex<f32>> {
// CHECK-DAG:         %[[IMAG_COEFF:.*]] = arith.constant dense<{{.*}}> : tensor<9x5xf32>
// CHECK-DAG:         %[[C0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:         %[[REAL_COEFF:.*]] = arith.constant dense<{{.*}}> : tensor<9x5xf32>
// CHECK-DAG:         %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[INPUT_VTENSOR:.*]] : !torch.vtensor<[16,9],f32> -> tensor<16x9xf32>
// CHECK:             %[[EMPTY_0:.*]] = tensor.empty() : tensor<16x5xf32>
// CHECK:             %[[ZEROES_0:.*]] = linalg.fill ins(%[[C0:.*]] : f32) outs(%[[EMPTY_0:.*]] : tensor<16x5xf32>) -> tensor<16x5xf32>
// CHECK:             %[[REAL_COMP:.*]] = linalg.matmul ins(%[[INPUT:.*]], %[[REAL_COEFF:.*]] : tensor<16x9xf32>, tensor<9x5xf32>) outs(%[[ZEROES_0:.*]] : tensor<16x5xf32>) -> tensor<16x5xf32>
// CHECK:             %[[EMPTY_1:.*]] = tensor.empty() : tensor<16x5xf32>
// CHECK:             %[[ZEROES_1:.*]] = linalg.fill ins(%[[C0:.*]] : f32) outs(%[[EMPTY_1:.*]] : tensor<16x5xf32>) -> tensor<16x5xf32>
// CHECK:             %[[IMAG_COMP:.*]] = linalg.matmul ins(%[[INPUT:.*]], %[[IMAG_COEFF:.*]] : tensor<16x9xf32>, tensor<9x5xf32>) outs(%[[ZEROES_1:.*]] : tensor<16x5xf32>) -> tensor<16x5xf32>
// CHECK:             %[[EMPTY_2:.*]] = tensor.empty() : tensor<16x5xcomplex<f32>>
// CHECK:             %[[COMPLEX:.*]] = linalg.generic {{.*}} ins(%[[REAL_COMP:.*]], %[[IMAG_COMP:.*]] : tensor<16x5xf32>, tensor<16x5xf32>) outs(%[[EMPTY_2:.*]] : tensor<16x5xcomplex<f32>>) {
// CHECK:             ^bb0(%[[IN:.*]]: f32, %[[IN_2:.*]]: f32, %[[OUT:.*]]: complex<f32>):
// CHECK:                %[[ELEM_COMPLEX:.*]] = complex.create %[[IN:.*]], %[[IN_2:.*]] : complex<f32>
// CHECK:                linalg.yield %[[ELEM_COMPLEX:.*]] : complex<f32>
// CHECK:             } -> tensor<16x5xcomplex<f32>>
// CHECK:             %[[OUTPUT_VTENSOR:.*]] = torch_c.from_builtin_tensor %[[COMPLEX:.*]] : tensor<16x5xcomplex<f32>> -> !torch.vtensor<[16,5],complex<f32>>
// CHECK:             return %[[OUTPUT_VTENSOR:.*]] : !torch.vtensor<[16,5],complex<f32>>

func.func @torch.aten.fft_rfft$2d_last_dim(%arg0: !torch.vtensor<[16,9],f32>) -> !torch.vtensor<[16,5],complex<f32>> {
  %int-1 = torch.constant.int -1
  %none = torch.constant.none
  %out = torch.aten.fft_rfft %arg0, %none, %int-1, %none : !torch.vtensor<[16,9],f32>, !torch.none, !torch.int, !torch.none -> !torch.vtensor<[16,5],complex<f32>>
  return %out : !torch.vtensor<[16,5],complex<f32>>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.fft_rfft$2d_first_dim(
// CHECK-SAME:           %[[INPUT_VTENSOR:.*]]: !torch.vtensor<[36,23],f32>) -> !torch.vtensor<[19,23],complex<f32>> {
// CHECK-DAG:         %[[IMAG_COEFF:.*]] = arith.constant dense<{{.*}}> : tensor<36x19xf32>
// CHECK-DAG:         %[[C0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:         %[[REAL_COEFF:.*]] = arith.constant dense<{{.*}}> : tensor<36x19xf32>
// CHECK-DAG:         %[[INPUT:.*]] = torch_c.to_builtin_tensor %[[INPUT_VTENSOR:.*]] : !torch.vtensor<[36,23],f32> -> tensor<36x23xf32>
// CHECK-DAG:         %[[EMPTY_0:.*]] = tensor.empty() : tensor<23x36xf32>
// CHECK:             %[[TRANSPOSED:.*]] = linalg.transpose ins(%[[INPUT:.*]] : tensor<36x23xf32>) outs(%[[EMPTY_0:.*]] : tensor<23x36xf32>) permutation = [1, 0]
// CHECK:             %[[EMPTY_1:.*]] = tensor.empty() : tensor<23x19xf32>
// CHECK:             %[[ZEROES_0:.*]] = linalg.fill ins(%[[C0:.*]] : f32) outs(%[[EMPTY_1:.*]] : tensor<23x19xf32>) -> tensor<23x19xf32>
// CHECK:             %[[REAL_COMP:.*]] = linalg.matmul ins(%[[TRANSPOSED:.*]], %[[REAL_COEFF:.*]] : tensor<23x36xf32>, tensor<36x19xf32>) outs(%[[ZEROES_0:.*]] : tensor<23x19xf32>) -> tensor<23x19xf32>
// CHECK:             %[[EMPTY_2:.*]] = tensor.empty() : tensor<23x19xf32>
// CHECK:             %[[ZEROES_1:.*]] = linalg.fill ins(%[[C0:.*]] : f32) outs(%[[EMPTY_2:.*]] : tensor<23x19xf32>) -> tensor<23x19xf32>
// CHECK:             %[[IMAG_COMP:.*]] = linalg.matmul ins(%[[TRANSPOSED:.*]], %[[IMAG_COEFF:.*]] : tensor<23x36xf32>, tensor<36x19xf32>) outs(%[[ZEROES_1:.*]] : tensor<23x19xf32>) -> tensor<23x19xf32>
// CHECK:             %[[EMPTY_3:.*]] = tensor.empty() : tensor<23x19xcomplex<f32>>
// CHECK:             %[[COMPLEX:.*]] = linalg.generic {{.*}} ins(%[[REAL_COMP:.*]], %[[IMAG_COMP:.*]] : tensor<23x19xf32>, tensor<23x19xf32>) outs(%[[EMPTY_3:.*]] : tensor<23x19xcomplex<f32>>) {
// CHECK:             ^bb0(%[[IN:.*]]: f32, %[[IN_3:.*]]: f32, %[[OUT:.*]]: complex<f32>):
// CHECK:                %[[EMPTY_02:.*]] = complex.create %[[IN:.*]], %[[IN_3:.*]] : complex<f32>
// CHECK:                linalg.yield %[[EMPTY_02:.*]] : complex<f32>
// CHECK:             } -> tensor<23x19xcomplex<f32>>
// CHECK:             %[[EMPTY_4:.*]] = tensor.empty() : tensor<19x23xcomplex<f32>>
// CHECK:             %[[TRANSPOSED_2:.*]] = linalg.transpose ins(%[[COMPLEX:.*]] : tensor<23x19xcomplex<f32>>) outs(%[[EMPTY_4:.*]] : tensor<19x23xcomplex<f32>>) permutation = [1, 0]
// CHECK:             %[[OUTPUT_VTENSOR:.*]] = torch_c.from_builtin_tensor %[[TRANSPOSED_2:.*]] : tensor<19x23xcomplex<f32>> -> !torch.vtensor<[19,23],complex<f32>>
// CHECK:             return %[[OUTPUT_VTENSOR:.*]] : !torch.vtensor<[19,23],complex<f32>>
func.func @torch.aten.fft_rfft$2d_first_dim(%arg0: !torch.vtensor<[36,23],f32>) -> !torch.vtensor<[19,23],complex<f32>> {
  %int0 = torch.constant.int 0
  %none = torch.constant.none
  %out = torch.aten.fft_rfft %arg0, %none, %int0, %none : !torch.vtensor<[36,23],f32>, !torch.none, !torch.int, !torch.none -> !torch.vtensor<[19,23],complex<f32>>
  return %out : !torch.vtensor<[19,23],complex<f32>>
}
