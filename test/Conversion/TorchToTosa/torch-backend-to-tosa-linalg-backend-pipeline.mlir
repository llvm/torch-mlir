// RUN: torch-mlir-opt -pass-pipeline='builtin.module(torch-backend-to-tosa-linalg-backend-pipeline)' -split-input-file -verify-diagnostics %s | FileCheck %s

//-----

// CHECK-LABEL:   func.func @torch.aten.size.int(
// CHECK-SAME:                                   %[[ARG0:.*]]: tensor<4x2xf32>) -> i64 {
// CHECK:           %[[VAL_0:.*]] = arith.constant false
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : index
// CHECK:           cf.assert %[[VAL_0]], "dim must be smaller than inputRank"
// CHECK:           %[[VAL_2:.*]] = tensor.dim %[[ARG0]], %[[VAL_1]] : tensor<4x2xf32>
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_2]] : index to i64
// CHECK:           return %[[VAL_3]] : i64
func.func @torch.aten.size.int(%arg0: !torch.vtensor<[4,2],f32>) -> !torch.int {
    %c2 = torch.constant.int 2
    %0 = torch.aten.size.int %arg0, %c2 : !torch.vtensor<[4,2],f32>, !torch.int -> !torch.int
    return %0 : !torch.int
}
