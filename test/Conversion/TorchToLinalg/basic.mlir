// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s
// RUN: torch-mlir-opt <%s -convert-torch-to-linalg=allow-non-finites=false -split-input-file -verify-diagnostics | FileCheck %s -check-prefix=UNSUPPORTED-NON-FINITES


// CHECK-LABEL:   func.func @torch.aten.mm$basic(
// CHECK-SAME:                        %[[LHS_VTENSOR:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                        %[[RHS_VTENSOR:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,2],f32> {
// CHECK-DAG:       %[[LHS:.*]] = torch_c.to_builtin_tensor %[[LHS_VTENSOR]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK-DAG:       %[[RHS:.*]] = torch_c.to_builtin_tensor %[[RHS_VTENSOR]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[LHS_DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[RHS_DIM_1:.*]] = tensor.dim %[[RHS]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[LHS_DIM_1:.*]] = tensor.dim %[[LHS]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[RHS_DIM_0:.*]] = tensor.dim %[[RHS]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[EQ:.*]] = arith.cmpi eq, %[[LHS_DIM_1]], %[[RHS_DIM_0]] : index
// CHECK:           assert %[[EQ]], "mismatching contracting dimension for torch.aten.mm"
// CHECK:           %[[INIT_TENSOR:.*]] = tensor.empty(%[[LHS_DIM_0]], %[[RHS_DIM_1]]) : tensor<?x?xf32>
// CHECK:           %[[CF0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[ZEROFILL:.*]] = linalg.fill ins(%[[CF0]] : f32) outs(%[[INIT_TENSOR]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[MATMUL:.*]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ZEROFILL]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[CASTED:.*]] = tensor.cast %[[MATMUL]] : tensor<?x?xf32> to tensor<?x2xf32>
// CHECK:           %[[RESULT_VTENSOR:.*]] = torch_c.from_builtin_tensor %[[CASTED]] : tensor<?x2xf32> -> !torch.vtensor<[?,2],f32>
// CHECK:           return %[[RESULT_VTENSOR]] : !torch.vtensor<[?,2],f32>
func.func @torch.aten.mm$basic(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,2],f32> {
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,2],f32>
  return %0 : !torch.vtensor<[?,2],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.matmul.2d
func.func @torch.aten.matmul.2d(%arg0: !torch.vtensor<[8,16],f32>, %arg1: !torch.vtensor<[16,8],f32>) -> !torch.vtensor<[8,8],f32> {
  // CHECK-DAG:  %[[LHS:.+]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[8,16],f32> -> tensor<8x16xf32>
  // CHECK-DAG:  %[[RHS:.+]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[16,8],f32> -> tensor<16x8xf32>
  // CHECK-DAG:  %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG:  %[[EMPTY:.+]] = tensor.empty() : tensor<8x8xf32>
  // CHECK-DAG:  %[[FILL:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK:  %[[MATMUL:.+]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<8x16xf32>, tensor<16x8xf32>) outs(%[[FILL]] : tensor<8x8xf32>) -> tensor<8x8xf32>
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[8,16],f32>, !torch.vtensor<[16,8],f32> -> !torch.vtensor<[8,8],f32>
  return %0 : !torch.vtensor<[8,8],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.matmul.4d
// CHECK-DAG:  %[[LHS:.+]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[1,2,32,400],f32> -> tensor<1x2x32x400xf32>
// CHECK-DAG:  %[[RHS:.+]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,2,400,32],f32> -> tensor<1x2x400x32xf32>
// CHECK-DAG:  %[[LHS_CAST:.*]] = tensor.cast %[[LHS]] : tensor<1x2x32x400xf32> to tensor<1x2x32x400xf32>
// CHECK-DAG:  %[[RHS_CAST:.*]] = tensor.cast %[[RHS]] : tensor<1x2x400x32xf32> to tensor<1x2x400x32xf32>
// CHECK-DAG:  %[[COLLAPSED_LHS:.+]] = tensor.collapse_shape %[[LHS_CAST]] {{\[\[}}0, 1], [2], [3]] : tensor<1x2x32x400xf32> into tensor<2x32x400xf32>
// CHECK-DAG:  %[[COLLAPSED_RHS:.+]] = tensor.collapse_shape %[[RHS_CAST]] {{\[\[}}0, 1], [2], [3]] : tensor<1x2x400x32xf32> into tensor<2x400x32xf32>
// CHECK:  %[[MATMUL:.+]] = linalg.batch_matmul ins(%[[COLLAPSED_RHS]], %[[COLLAPSED_LHS]] : tensor<2x400x32xf32>, tensor<2x32x400xf32>) outs(%{{.*}} : tensor<2x400x400xf32>) -> tensor<2x400x400xf32>
// CHECK:  %[[EXPANDED:.+]] = tensor.expand_shape %[[MATMUL]] {{\[\[}}0, 1], [2], [3]] output_shape [1, 2, 400, 400] : tensor<2x400x400xf32> into tensor<1x2x400x400xf32>
func.func @torch.aten.matmul.4d(%arg0: !torch.vtensor<[1,2,32,400],f32>, %arg1: !torch.vtensor<[1,2,400,32],f32>) -> !torch.vtensor<[1,2,400,400],f32> {
  %0 = torch.aten.matmul %arg1, %arg0 : !torch.vtensor<[1,2,400,32],f32>, !torch.vtensor<[1,2,32,400],f32> -> !torch.vtensor<[1,2,400,400],f32>
  return %0 : !torch.vtensor<[1,2,400,400],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.matmul$dot_f16
// CHECK:      %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:      linalg.dot ins(%{{.*}}, %{{.*}} : tensor<4xf16>, tensor<4xf16>) outs(%{{.*}} : tensor<f32>) -> tensor<f32>
// CHECK:      arith.truncf %{{.*}} : f32 to f16
func.func @torch.aten.matmul$dot_f16(%arg0: !torch.vtensor<[4],f16>, %arg1: !torch.vtensor<[4],f16>) -> !torch.vtensor<[],f16> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[4],f16>, !torch.vtensor<[4],f16> -> !torch.vtensor<[],f16>
  return %0 : !torch.vtensor<[],f16>
}

// -----

// CHECK-LABEL: func.func @torch.aten.matmul$vecmat_f16
// CHECK:      %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:      linalg.vecmat ins(%{{.*}}, %{{.*}} : tensor<4xf16>, tensor<4x8xf16>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK:      arith.truncf %{{.*}} : f32 to f16
func.func @torch.aten.matmul$vecmat_f16(%arg0: !torch.vtensor<[4],f16>, %arg1: !torch.vtensor<[4,8],f16>) -> !torch.vtensor<[8],f16> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[4],f16>, !torch.vtensor<[4,8],f16> -> !torch.vtensor<[8],f16>
  return %0 : !torch.vtensor<[8],f16>
}

// -----

// CHECK-LABEL: func.func @torch.aten.matmul$matvec_f16
// CHECK:      %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:      linalg.matvec ins(%{{.*}}, %{{.*}} : tensor<8x4xf16>, tensor<4xf16>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK:      arith.truncf %{{.*}} : f32 to f16
func.func @torch.aten.matmul$matvec_f16(%arg0: !torch.vtensor<[8,4],f16>, %arg1: !torch.vtensor<[4],f16>) -> !torch.vtensor<[8],f16> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[8,4],f16>, !torch.vtensor<[4],f16> -> !torch.vtensor<[8],f16>
  return %0 : !torch.vtensor<[8],f16>
}

// -----

// CHECK-LABEL: func.func @torch.aten.matmul$2d_f16
// CHECK:      %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:      linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<8x16xf16>, tensor<16x8xf16>) outs(%{{.*}} : tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:      arith.truncf %{{.*}} : f32 to f16
func.func @torch.aten.matmul$2d_f16(%arg0: !torch.vtensor<[8,16],f16>, %arg1: !torch.vtensor<[16,8],f16>) -> !torch.vtensor<[8,8],f16> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[8,16],f16>, !torch.vtensor<[16,8],f16> -> !torch.vtensor<[8,8],f16>
  return %0 : !torch.vtensor<[8,8],f16>
}

// -----

// CHECK-LABEL: func.func @torch.aten.matmul$3d_f16
// CHECK:      %[[BMM:.+]] = linalg.batch_matmul ins(%{{.*}}, %{{.*}} : tensor<2x8x16xf16>, tensor<2x16x8xf16>) outs(%{{.*}} : tensor<2x8x8xf32>) -> tensor<2x8x8xf32>
// CHECK:      arith.truncf %{{.*}} : f32 to f16
func.func @torch.aten.matmul$3d_f16(%arg0: !torch.vtensor<[2,8,16],f16>, %arg1: !torch.vtensor<[2,16,8],f16>) -> !torch.vtensor<[2,8,8],f16> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[2,8,16],f16>, !torch.vtensor<[2,16,8],f16> -> !torch.vtensor<[2,8,8],f16>
  return %0 : !torch.vtensor<[2,8,8],f16>
}

// -----

// CHECK-LABEL: func.func @torch.aten.matmul$4d_f16
// CHECK:      %[[COLLAPSED_LHS:.+]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1], [2], [3]] : tensor<1x2x8x16xf16> into tensor<2x8x16xf16>
// CHECK:      %[[COLLAPSED_RHS:.+]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1], [2], [3]] : tensor<1x2x16x8xf16> into tensor<2x16x8xf16>
// CHECK:      %[[BMM:.+]] = linalg.batch_matmul ins(%[[COLLAPSED_LHS]], %[[COLLAPSED_RHS]] : tensor<2x8x16xf16>, tensor<2x16x8xf16>) outs(%{{.*}} : tensor<2x8x8xf32>) -> tensor<2x8x8xf32>
// CHECK:      arith.truncf %{{.*}} : f32 to f16
// CHECK:      tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2], [3]] output_shape [1, 2, 8, 8] : tensor<2x8x8xf16> into tensor<1x2x8x8xf16>
func.func @torch.aten.matmul$4d_f16(%arg0: !torch.vtensor<[1,2,8,16],f16>, %arg1: !torch.vtensor<[1,2,16,8],f16>) -> !torch.vtensor<[1,2,8,8],f16> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[1,2,8,16],f16>, !torch.vtensor<[1,2,16,8],f16> -> !torch.vtensor<[1,2,8,8],f16>
  return %0 : !torch.vtensor<[1,2,8,8],f16>
}

// -----

// CHECK-LABEL: func.func @torch.aten.matmul$2d_bf16
// CHECK:      %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:      linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<8x16xbf16>, tensor<16x8xbf16>) outs(%{{.*}} : tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:      arith.truncf %{{.*}} : f32 to bf16
func.func @torch.aten.matmul$2d_bf16(%arg0: !torch.vtensor<[8,16],bf16>, %arg1: !torch.vtensor<[16,8],bf16>) -> !torch.vtensor<[8,8],bf16> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[8,16],bf16>, !torch.vtensor<[16,8],bf16> -> !torch.vtensor<[8,8],bf16>
  return %0 : !torch.vtensor<[8,8],bf16>
}

// -----

// CHECK-LABEL: func.func @torch.aten.matmul$generic_f16
// CHECK:      linalg.generic
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<?x?x8x16xf16>, tensor<?x?x16x8xf16>)
// CHECK-SAME: outs(%{{.*}} : tensor<?x?x8x8xf32>)
// CHECK:      ^bb0(%[[A:.+]]: f16, %[[B:.+]]: f16, %[[C:.+]]: f32):
// CHECK:        %[[EA:.+]] = arith.extf %[[A]] : f16 to f32
// CHECK:        %[[EB:.+]] = arith.extf %[[B]] : f16 to f32
// CHECK:        %[[MUL:.+]] = arith.mulf %[[EA]], %[[EB]] : f32
// CHECK:        %[[ADD:.+]] = arith.addf %[[MUL]], %[[C]] : f32
// CHECK:        linalg.yield %[[ADD]] : f32
// CHECK:      arith.truncf %{{.*}} : f32 to f16
func.func @torch.aten.matmul$generic_f16(%arg0: !torch.vtensor<[?,?,8,16],f16>, %arg1: !torch.vtensor<[?,?,16,8],f16>) -> !torch.vtensor<[?,?,8,8],f16> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?,8,16],f16>, !torch.vtensor<[?,?,16,8],f16> -> !torch.vtensor<[?,?,8,8],f16>
  return %0 : !torch.vtensor<[?,?,8,8],f16>
}

// -----

// Verify that aten.bmm(si8, si8) -> si32 uses i32 accumulator (not i64),
// enabling downstream i8 HW intrinsics.
// CHECK-LABEL: func.func @torch.aten.bmm$i8
// CHECK:      linalg.batch_matmul ins(%{{.*}}, %{{.*}} : tensor<2x8x16xi8>, tensor<2x16x8xi8>) outs(%{{.*}} : tensor<2x8x8xi32>) -> tensor<2x8x8xi32>
// CHECK-NOT:  arith.extsi
// CHECK-NOT:  arith.trunci
func.func @torch.aten.bmm$i8(%arg0: !torch.vtensor<[2,8,16],si8>, %arg1: !torch.vtensor<[2,16,8],si8>) -> !torch.vtensor<[2,8,8],si32> {
  %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[2,8,16],si8>, !torch.vtensor<[2,16,8],si8> -> !torch.vtensor<[2,8,8],si32>
  return %0 : !torch.vtensor<[2,8,8],si32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.bmm$mixed_si4_f16
// CHECK:      linalg.batch_matmul ins(%{{.*}}, %{{.*}} : tensor<1x4x8xf16>, tensor<1x8x4xf16>) outs(%{{.*}} : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
func.func @torch.aten.bmm$mixed_si4_f16(%arg0: !torch.vtensor<[1,4,8],si4>, %arg1: !torch.vtensor<[1,8,4],f16>) -> !torch.vtensor<[1,4,4],f16> {
  %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[1,4,8],si4>, !torch.vtensor<[1,8,4],f16> -> !torch.vtensor<[1,4,4],f16>
  return %0 : !torch.vtensor<[1,4,4],f16>
}

// -----

// CHECK-LABEL: func.func @torch.aten.mm$basic_strict(
// CHECK-NOT: assert
func.func @torch.aten.mm$basic_strict(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,2],f32>
  attributes {torch.assume_strict_symbolic_shapes}
{
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,2],f32>
  return %0 : !torch.vtensor<[?,2],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.mm$basic_unsigned(
// CHECK: linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}
func.func @torch.aten.mm$basic_unsigned(%arg0: !torch.vtensor<[?,?],ui32>, %arg1: !torch.vtensor<[?,?],ui32>) -> !torch.vtensor<[?,2],ui32>
  attributes {torch.assume_strict_symbolic_shapes}
{
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?,?],ui32>, !torch.vtensor<[?,?],ui32> -> !torch.vtensor<[?,2],ui32>
  return %0 : !torch.vtensor<[?,2],ui32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.mm$mixed_signedness_lhs_unsigned(
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<3x4xi8>, tensor<4x3xi8>)
// CHECK: ^bb0(%[[LHS:.*]]: i8, %[[RHS:.*]]: i8, %[[ACC:.*]]: i32):
// CHECK-DAG: %[[LHS_EXT:.*]] = arith.extui %[[LHS]] : i8 to i32
// CHECK-DAG: %[[RHS_EXT:.*]] = arith.extsi %[[RHS]] : i8 to i32
// CHECK: %[[MUL:.*]] = arith.muli %[[LHS_EXT]], %[[RHS_EXT]] : i32
// CHECK: %[[SUM:.*]] = arith.addi %[[ACC]], %[[MUL]] : i32
// CHECK: linalg.yield %[[SUM]] : i32
func.func @torch.aten.mm$mixed_signedness_lhs_unsigned(%arg0: !torch.vtensor<[3,4],ui8>, %arg1: !torch.vtensor<[4,3],si8>) -> !torch.vtensor<[3,3],si32> {
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[3,4],ui8>, !torch.vtensor<[4,3],si8> -> !torch.vtensor<[3,3],si32>
  return %0 : !torch.vtensor<[3,3],si32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.mm$mixed_signedness_rhs_unsigned(
// CHECK: ^bb0(%[[LHS:.*]]: i8, %[[RHS:.*]]: i8, %[[ACC:.*]]: i32):
// CHECK-DAG: %[[LHS_EXT:.*]] = arith.extsi %[[LHS]] : i8 to i32
// CHECK-DAG: %[[RHS_EXT:.*]] = arith.extui %[[RHS]] : i8 to i32
// CHECK: arith.muli %[[LHS_EXT]], %[[RHS_EXT]] : i32
func.func @torch.aten.mm$mixed_signedness_rhs_unsigned(%arg0: !torch.vtensor<[3,4],si8>, %arg1: !torch.vtensor<[4,3],ui8>) -> !torch.vtensor<[3,3],si32> {
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[3,4],si8>, !torch.vtensor<[4,3],ui8> -> !torch.vtensor<[3,3],si32>
  return %0 : !torch.vtensor<[3,3],si32>
}

// -----

// If the operands are missing dtype, we cannot lower it.
func.func @torch.aten.mm$no_convert$missing_dtype(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> !torch.vtensor {
  // expected-error@+1 {{failed to legalize}}
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor, !torch.vtensor -> !torch.vtensor
  return %0 : !torch.vtensor
}

// -----

// Correctly handle the case that operands are statically the wrong rank
// (rank 1 vs rank 2 expected for matmul.)
func.func @torch.aten.mm$no_convert$wrong_rank(%arg0: !torch.vtensor<[?],f32>, %arg1: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?],f32> {
  // expected-error@+1 {{failed to legalize}}
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?],f32>, !torch.vtensor<[?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// If the result is missing dtype, we cannot lower it.
func.func @torch.aten.mm$no_convert$result_missing_dtype(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor {
  // expected-error@+1 {{failed to legalize}}
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.vtensor
  return %0 : !torch.vtensor
}

// -----

// CHECK-LABEL:     func.func @torch.aten.Int.Tensor$zero_rank
// CHECK-SAME:          (%[[ARG:.*]]: !torch.vtensor<[],si64>) -> !torch.int {
// CHECK:               %[[I:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[],si64> -> tensor<i64>
// CHECK:               %[[EXT:.*]] = tensor.extract %[[I]][] : tensor<i64>
// CHECK:               %[[RET:.*]] = torch_c.from_i64 %[[EXT]]
// CHECK:               return %[[RET]] : !torch.int
func.func @torch.aten.Int.Tensor$zero_rank(%arg0: !torch.vtensor<[],si64>) -> !torch.int {
  %0 = torch.aten.Int.Tensor %arg0 : !torch.vtensor<[],si64> -> !torch.int
  return %0 : !torch.int
}

// -----

// CHECK-LABEL:     func.func @torch.aten.Int.Tensor$non_zero_rank
// CHECK-SAME:          (%[[ARG:.*]]: !torch.vtensor<[?,?],si64>) -> !torch.int {
// CHECK:               %[[I:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
// CHECK:               %[[C0:.*]] = arith.constant 0 : index
// CHECK:               %[[DIM0:.*]] = tensor.dim %[[I]], %[[C0]] : tensor<?x?xi64>
// CHECK:               %[[C1:.*]] = arith.constant 1 : index
// CHECK:               %[[DIM1:.*]] = tensor.dim %[[I]], %[[C1]] : tensor<?x?xi64>
// CHECK:               %[[ONE:.*]] = arith.constant 1 : i64
// CHECK:               %[[DIM0_INDEX:.*]] = arith.index_cast %[[DIM0]] : index to i64
// CHECK:               %[[PRED0:.*]] = arith.cmpi eq, %[[DIM0_INDEX]], %[[ONE]] : i64
// CHECK:               assert %[[PRED0]], "mismatching contracting dimension"
// CHECK:               %[[DIM1_INDEX:.*]] = arith.index_cast %[[DIM1]] : index to i64
// CHECK:               %[[PRED1:.*]] = arith.cmpi eq, %[[DIM1_INDEX]], %[[ONE]] : i64
// CHECK:               assert %[[PRED1]], "mismatching contracting dimension"
// CHECK:               %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:               %[[EXT:.*]] = tensor.extract %[[I]][%[[ZERO]], %[[ZERO]]] : tensor<?x?xi64>
// CHECK:               %[[RET:.*]] = torch_c.from_i64 %[[EXT]]
// CHECK:               return %[[RET]] : !torch.int
func.func @torch.aten.Int.Tensor$non_zero_rank(%arg0: !torch.vtensor<[?,?],si64>) -> !torch.int {
  %0 = torch.aten.Int.Tensor %arg0 : !torch.vtensor<[?,?],si64> -> !torch.int
  return %0 : !torch.int
}

// -----

// CHECK-LABEL:     func.func @torch.aten.Int.Tensor$zero_rank$byte_dtype
// CHECK-SAME:          (%[[ARG:.*]]: !torch.vtensor<[],ui8>) -> !torch.int {
// CHECK:               %[[I:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[],ui8> -> tensor<i8>
// CHECK:               %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:               %[[C0:.*]] = arith.constant 0 : index
// CHECK:               %[[EXTRACT:.*]] = tensor.extract %[[I]][] : tensor<i8>
// CHECK:               %[[RES:.*]] = arith.extui %[[EXTRACT]] : i8 to i64
// CHECK:               %[[RET:.*]] = torch_c.from_i64 %[[RES]]
// CHECK:               return %[[RET]] : !torch.int
func.func @torch.aten.Int.Tensor$zero_rank$byte_dtype(%arg0: !torch.vtensor<[],ui8>) -> !torch.int {
  %0 = torch.aten.Int.Tensor %arg0 : !torch.vtensor<[],ui8> -> !torch.int
  return %0 : !torch.int
}

// -----

// CHECK-LABEL:     func.func @torch.aten.Int.Tensor$zero_rank$char_dtype
// CHECK-SAME:          (%[[ARG:.*]]: !torch.vtensor<[],si8>) -> !torch.int {
// CHECK:               %[[I:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[],si8> -> tensor<i8>
// CHECK:               %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:               %[[C0:.*]] = arith.constant 0 : index
// CHECK:               %[[EXTRACT:.*]] = tensor.extract %[[I]][] : tensor<i8>
// CHECK:               %[[RES:.*]] = arith.extsi %[[EXTRACT]] : i8 to i64
// CHECK:               %[[RET:.*]] = torch_c.from_i64 %[[RES]]
// CHECK:               return %[[RET]] : !torch.int
func.func @torch.aten.Int.Tensor$zero_rank$char_dtype(%arg0: !torch.vtensor<[],si8>) -> !torch.int {
  %0 = torch.aten.Int.Tensor %arg0 : !torch.vtensor<[],si8> -> !torch.int
  return %0 : !torch.int
}

// -----

// CHECK-LABEL:     func.func @torch.aten.Float.Tensor$zero_rank
// CHECK-SAME:          (%[[ARG:.*]]: !torch.vtensor<[],f64>) -> !torch.float {
// CHECK:               %[[F:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[],f64> -> tensor<f64>
// CHECK:               %[[EXT:.*]] = tensor.extract %[[F]][] : tensor<f64>
// CHECK:               %[[RET:.*]] = torch_c.from_f64 %[[EXT]]
// CHECK:               return %[[RET]] : !torch.float
func.func @torch.aten.Float.Tensor$zero_rank(%arg0: !torch.vtensor<[],f64>) -> !torch.float {
  %0 = torch.aten.Float.Tensor %arg0 : !torch.vtensor<[],f64> -> !torch.float
  return %0 : !torch.float
}

// -----

// CHECK-LABEL:     func.func @torch.aten.Float.Tensor$non_zero_rank
// CHECK-SAME:          (%[[ARG:.*]]: !torch.vtensor<[?,?],f64>) -> !torch.float {
// CHECK:               %[[F:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f64> -> tensor<?x?xf64>
// CHECK:               %[[C0:.*]] = arith.constant 0 : index
// CHECK:               %[[DIM0:.*]] = tensor.dim %[[F]], %[[C0]] : tensor<?x?xf64>
// CHECK:               %[[C1:.*]] = arith.constant 1 : index
// CHECK:               %[[DIM1:.*]] = tensor.dim %[[F]], %[[C1]] : tensor<?x?xf64>
// CHECK:               %[[ONE:.*]] = arith.constant 1 : i64
// CHECK:               %[[DIM0_INDEX:.*]] = arith.index_cast %[[DIM0]] : index to i64
// CHECK:               %[[PRED0:.*]] = arith.cmpi eq, %[[DIM0_INDEX]], %[[ONE]] : i64
// CHECK:               assert %[[PRED0]], "mismatching contracting dimension"
// CHECK:               %[[DIM1_INDEX:.*]] = arith.index_cast %[[DIM1]] : index to i64
// CHECK:               %[[PRED1:.*]] = arith.cmpi eq, %[[DIM1_INDEX]], %[[ONE]] : i64
// CHECK:               assert %[[PRED1]], "mismatching contracting dimension"
// CHECK:               %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:               %[[EXT:.*]] = tensor.extract %[[F]][%[[ZERO]], %[[ZERO]]] : tensor<?x?xf64>
// CHECK:               %[[RET:.*]] = torch_c.from_f64 %[[EXT]]
// CHECK:               return %[[RET]] : !torch.float
func.func @torch.aten.Float.Tensor$non_zero_rank(%arg0: !torch.vtensor<[?,?],f64>) -> !torch.float {
  %0 = torch.aten.Float.Tensor %arg0 : !torch.vtensor<[?,?],f64> -> !torch.float
  return %0 : !torch.float
}

// -----

// CHECK-LABEL:     func.func @torch.aten.Bool.Tensor$zero_rank
// CHECK-SAME:          (%[[ARG:.*]]: !torch.vtensor<[],i1>) -> !torch.bool {
// CHECK:               %[[B:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[],i1> -> tensor<i1>
// CHECK:               %[[EXT:.*]] = tensor.extract %[[B]][] : tensor<i1>
// CHECK:               %[[RES:.*]] = torch_c.from_i1 %[[EXT]]
// CHECK:               return %[[RES]] : !torch.bool
func.func @torch.aten.Bool.Tensor$zero_rank(%arg0: !torch.vtensor<[],i1>) -> !torch.bool {
  %0 = torch.aten.Bool.Tensor %arg0 : !torch.vtensor<[],i1> -> !torch.bool
  return %0 : !torch.bool
}

// -----

// CHECK-LABEL:     func.func @torch.aten.Bool.Tensor$non_zero_rank
// CHECK-SAME:          (%[[ARG:.*]]: !torch.vtensor<[?,?],i1>) -> !torch.bool {
// CHECK:               %[[B:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],i1> -> tensor<?x?xi1>
// CHECK:               %[[C0:.*]] = arith.constant 0 : index
// CHECK:               %[[DIM0:.*]] = tensor.dim %[[B]], %[[C0]] : tensor<?x?xi1>
// CHECK:               %[[C1:.*]] = arith.constant 1 : index
// CHECK:               %[[DIM1:.*]] = tensor.dim %[[B]], %[[C1]] : tensor<?x?xi1>
// CHECK:               %[[ONE:.*]] = arith.constant 1 : i64
// CHECK:               %[[DIM0_INDEX:.*]] = arith.index_cast %[[DIM0]] : index to i64
// CHECK:               %[[PRED0:.*]] = arith.cmpi eq, %[[DIM0_INDEX]], %[[ONE]] : i64
// CHECK:               assert %[[PRED0]], "mismatching contracting dimension"
// CHECK:               %[[DIM1_INDEX:.*]] = arith.index_cast %[[DIM1]] : index to i64
// CHECK:               %[[PRED1:.*]] = arith.cmpi eq, %[[DIM1_INDEX]], %[[ONE]] : i64
// CHECK:               assert %[[PRED1]], "mismatching contracting dimension"
// CHECK:               %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:               %[[EXT:.*]] = tensor.extract %[[I]][%[[ZERO]], %[[ZERO]]] : tensor<?x?xi1>
// CHECK:               %[[RET:.*]] = torch_c.from_i1 %[[EXT]]
// CHECK:               return %[[RET]] : !torch.bool
func.func @torch.aten.Bool.Tensor$non_zero_rank(%arg0: !torch.vtensor<[?,?],i1>) -> !torch.bool {
  %0 = torch.aten.Bool.Tensor %arg0 : !torch.vtensor<[?,?],i1> -> !torch.bool
  return %0 : !torch.bool
}

// -----

// CHECK:    func.func @torch.prim.NumToTensor.Scalar$basic(%[[IN:.*]]: !torch.int) -> !torch.vtensor<[],si64> {
// CHECK:      %[[INI64:.*]] = torch_c.to_i64 %[[IN]]
// CHECK:      %[[NEWVEC:.*]] = tensor.empty() : tensor<i64>
// CHECK:      %[[FILLVEC:.*]] = linalg.fill ins(%[[INI64]] : i64) outs(%[[NEWVEC]] : tensor<i64>) -> tensor<i64>
// CHECK:      %[[OUTVEC:.*]] = torch_c.from_builtin_tensor %[[FILLVEC]] : tensor<i64> -> !torch.vtensor<[],si64>
// CHECK:      return %[[OUTVEC]] : !torch.vtensor<[],si64>
func.func @torch.prim.NumToTensor.Scalar$basic(%arg0: !torch.int) -> !torch.vtensor<[],si64> {
  %0 = torch.prim.NumToTensor.Scalar %arg0 : !torch.int -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}

// -----

// CHECK-LABEL: func.func @torch.aten.full$bf16$f_to_f
// CHECK:       %[[F64:.*]] = torch_c.to_f64 %{{.*}}
// CHECK:       %[[TRUNC:.*]] = arith.truncf %[[F64]] : f64 to bf16
// CHECK:       %[[EMPTY:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?xbf16>
// CHECK:       %[[FILLED:.*]] = linalg.fill ins(%[[TRUNC]] : bf16) outs(%[[EMPTY]] : tensor<?x?xbf16>) -> tensor<?x?xbf16>
func.func @torch.aten.full$bf16$f_to_f(%dim0: !torch.int, %dim1: !torch.int, %fill: !torch.float) -> !torch.vtensor<[?,?],bf16> {
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %dim0, %dim1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.full %0, %fill, %none, %none, %none, %none : !torch.list<int>, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?],bf16>
  return %1 : !torch.vtensor<[?,?],bf16>
}

// -----

// CHECK-LABEL: func.func @torch.aten.full$bf16$si_to_f
// CHECK:       %[[SITOFP:.*]] = arith.sitofp %{{.*}} : i64 to bf16
// CHECK:       %[[EMPTY:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?xbf16>
// CHECK:       %[[FILLED:.*]] = linalg.fill ins(%[[SITOFP]] : bf16) outs(%[[EMPTY]] : tensor<?x?xbf16>) -> tensor<?x?xbf16>
func.func @torch.aten.full$bf16$si_to_f(%dim0: !torch.int, %dim1: !torch.int, %fill: !torch.int) -> !torch.vtensor<[?,?],bf16> {
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %dim0, %dim1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.full %0, %fill, %none, %none, %none, %none : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?],bf16>
  return %1 : !torch.vtensor<[?,?],bf16>
}

// -----

// CHECK-LABEL: func.func @torch.aten.full$si32$f_to_si
// CHECK:       %[[F64:.*]] = torch_c.to_f64 %{{.*}}
// CHECK:       %[[FPTOSI:.*]] = arith.fptosi %[[F64]] : f64 to i32
// CHECK:       %[[EMPTY:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?xi32>
// CHECK:       %[[FILLED:.*]] = linalg.fill ins(%[[FPTOSI]] : i32) outs(%[[EMPTY]] : tensor<?x?xi32>) -> tensor<?x?xi32>
func.func @torch.aten.full$si32$f_to_si(%dim0: !torch.int, %dim1: !torch.int, %fill: !torch.float) -> !torch.vtensor<[?,?],si32> {
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %dim0, %dim1 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.full %0, %fill, %none, %none, %none, %none : !torch.list<int>, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[?,?],si32>
  return %1 : !torch.vtensor<[?,?],si32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.full$f32$truncf
// CHECK:       %[[F64:.*]] = torch_c.to_f64 %{{.*}}
// CHECK:       %[[TRUNC:.*]] = arith.truncf %[[F64]] : f64 to f32
// CHECK:       %[[EMPTY:.*]] = tensor.empty() : tensor<2048x7168xf32>
// CHECK:       %[[FILLED:.*]] = linalg.fill ins(%[[TRUNC]] : f32) outs(%[[EMPTY]] : tensor<2048x7168xf32>) -> tensor<2048x7168xf32>
func.func @torch.aten.full$f32$truncf() -> !torch.vtensor<[2048,7168],f32> {
  %float0.000000e00 = torch.constant.float 0.000000e+00
  %none = torch.constant.none
  %int2048 = torch.constant.int 2048
  %int7168 = torch.constant.int 7168
  %0 = torch.prim.ListConstruct %int2048, %int7168 : (!torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.aten.full %0, %float0.000000e00, %none, %none, %none, %none : !torch.list<int>, !torch.float, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2048,7168],f32>
  return %1 : !torch.vtensor<[2048,7168],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.tensor_static_info_cast$basic(
// CHECK-SAME:                                              %[[VALUE_T:.*]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[4],f32> {
// CHECK:           %[[T:.*]] = torch_c.to_builtin_tensor %[[VALUE_T]] : !torch.vtensor<[?],f32> -> tensor<?xf32>
// CHECK:           %[[T_CAST:.*]] = tensor.cast %[[T]] : tensor<?xf32> to tensor<4xf32>
// CHECK:           %[[VALUE_T_CAST:.*]] = torch_c.from_builtin_tensor %[[T_CAST]] : tensor<4xf32> -> !torch.vtensor<[4],f32>
// CHECK:           return %[[VALUE_T_CAST]] : !torch.vtensor<[4],f32>
func.func @torch.tensor_static_info_cast$basic(%t: !torch.vtensor<[?],f32>) -> !torch.vtensor<[4],f32> {
  %t_cast = torch.tensor_static_info_cast %t : !torch.vtensor<[?],f32> to !torch.vtensor<[4],f32>
  return %t_cast : !torch.vtensor<[4],f32>
}

// -----

// CHECK-LABEL:     func.func @torch.aten.neg
// CHECK: linalg.generic {{.*}} {
// CHECK-NEXT:    ^bb0(%[[LHS:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      %[[NEG:.*]] = arith.negf %[[LHS]] : f32
// CHECK-NEXT:      linalg.yield %[[NEG]] : f32
// CHECK-NEXT:    } -> tensor<?x?xf32>
func.func @torch.aten.neg(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.neg %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:     func.func @torch.aten.neg.bf16
// CHECK: linalg.generic {{.*}} {
// CHECK-NEXT:    ^bb0(%[[LHS:.*]]: bf16, %{{.*}}: bf16):
// CHECK-NEXT:      %[[NEG:.*]] = arith.negf %[[LHS]] : bf16
// CHECK-NEXT:      linalg.yield %[[NEG]] : bf16
// CHECK-NEXT:    } -> tensor<?x?xbf16>
func.func @torch.aten.neg.bf16(%arg0: !torch.vtensor<[?,?],bf16>) -> !torch.vtensor<[?,?],bf16> {
  %0 = torch.aten.neg %arg0 : !torch.vtensor<[?,?],bf16> -> !torch.vtensor<[?,?],bf16>
  return %0 : !torch.vtensor<[?,?],bf16>
}

// -----

// CHECK-LABEL:     func.func @torch.aten.neg.f16
// CHECK: linalg.generic {{.*}} {
// CHECK-NEXT:    ^bb0(%[[LHS:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT:      %[[NEG:.*]] = arith.negf %[[LHS]] : f16
// CHECK-NEXT:      linalg.yield %[[NEG]] : f16
// CHECK-NEXT:    } -> tensor<?x?xf16>
func.func @torch.aten.neg.f16(%arg0: !torch.vtensor<[?,?],f16>) -> !torch.vtensor<[?,?],f16> {
  %0 = torch.aten.neg %arg0 : !torch.vtensor<[?,?],f16> -> !torch.vtensor<[?,?],f16>
  return %0 : !torch.vtensor<[?,?],f16>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.cat$convert(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>, %[[ARG1:.*]]: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[INT0:.*]] = torch.constant.int 0
// CHECK:         %[[T0:.*]] = torch.prim.ListConstruct %[[ARG0]], %[[ARG1]] : (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],si32>) -> !torch.list<vtensor>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:         %[[T2:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?,?],si32> -> tensor<?x?xi32>
// CHECK:         %[[T3:.*]] = linalg.generic {{.*}} ins(%[[T2]] : tensor<?x?xi32>) outs(%{{.*}}: tensor<?x?xf32>)
// CHECK:         %[[T4:.*]] = tensor.concat dim(0) %[[T1]], %[[T3]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[T5:.*]] = torch_c.from_builtin_tensor %[[T4]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[T5]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.cat$convert(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],si32>) -> !torch.list<vtensor>
  %1 = torch.aten.cat %0, %int0 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %1 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.cat(
// CHECK-SAME:                              %[[ARG_0:.*]]: !torch.vtensor<[?,?],f32>,
// CHECK-SAME:                              %[[ARG_1:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %[[VAL_0:.*]] = torch.prim.ListConstruct %[[ARG_0]], %[[ARG_1]] : (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>) -> !torch.list<vtensor>
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[ARG_1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.concat dim(0) %[[VAL_1]], %[[VAL_2]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.cat(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>) -> !torch.list<vtensor>
  %1 = torch.aten.cat %0, %int0 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %1 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.transpose$basic(
// CHECK-SAME:                                     %[[ARG_0:.*]]: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[3,4],f32> {
// CHECK:           %[[IN_0:.*]] = torch_c.to_builtin_tensor %[[ARG_0]] : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
// CHECK:           %[[TRANSP:.*]] = linalg.transpose ins(%[[IN_0]] : tensor<4x3xf32>) outs(%1 : tensor<3x4xf32>) permutation = [1, 0]
// CHECK:           %[[OUT_0:.*]] = torch_c.from_builtin_tensor %{{.*}}  : tensor<3x4xf32> -> !torch.vtensor<[3,4],f32>
// CHECK:           return %[[OUT_0]] : !torch.vtensor<[3,4],f32>
func.func @torch.aten.transpose$basic(%arg0: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[3,4],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.aten.transpose.int %arg0, %int0, %int1 : !torch.vtensor<[4,3],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// A rank-4 input (batch, num_heads, seq, head_size) is rotated with
// position_ids indexing dims 0 and 2.
// CHECK-DAG: #[[$ID:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$POS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK-LABEL:   func.func @test_rotary_embedding(
// CHECK:           %[[SIN:.*]] = torch_c.to_builtin_tensor %arg3 : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
// CHECK:           %[[COS:.*]] = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[1,3,2,6],f32> -> tensor<1x3x2x6xf32>
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<1x3x2x6xf32>
// CHECK-DAG:       %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       %[[MINUS_ONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG:       %[[ROT_DIM:.*]] = arith.constant 6 : index
// CHECK-DAG:       %[[HALF_ROT_DIM:.*]] = arith.constant 3 : index
// CHECK:           %[[ROTARY:.*]] = linalg.generic {indexing_maps = [#[[$ID]], #[[$POS]], #[[$ID]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[INPUT]], %{{.*}} : tensor<1x3x2x6xf32>, tensor<1x2xi64>) outs(%[[EMPTY]] : tensor<1x3x2x6xf32>) {
// CHECK:           ^bb0(%[[IN:.*]]: f32, %[[POS_ID:.*]]: i64, %{{.*}}: f32):
// CHECK:             %[[I0:.*]] = linalg.index 0 : index
// CHECK:             %[[I1:.*]] = linalg.index 1 : index
// CHECK:             %[[I2:.*]] = linalg.index 2 : index
// CHECK:             %[[H:.*]] = linalg.index 3 : index
// CHECK:             %[[CACHE_IDX:.*]] = arith.remsi %[[H]], %[[HALF_ROT_DIM]] : index
// CHECK:             %[[SIGN:.*]] = arith.cmpi sge, %[[H]], %[[HALF_ROT_DIM]] : index
// CHECK:             %[[SHIFTED:.*]] = arith.addi %[[H]], %[[HALF_ROT_DIM]] : index
// CHECK:             %[[J:.*]] = arith.remsi %[[SHIFTED]], %[[ROT_DIM]] : index
// CHECK:             %[[POS_IDX:.*]] = arith.index_cast %[[POS_ID]] : i64 to index
// CHECK:             %[[COS_EMB:.*]] = tensor.extract %[[COS]]{{\[}}%[[POS_IDX]], %[[CACHE_IDX]]] : tensor<4x3xf32>
// CHECK:             %[[SIN_EMB:.*]] = tensor.extract %[[SIN]]{{\[}}%[[POS_IDX]], %[[CACHE_IDX]]] : tensor<4x3xf32>
// CHECK:             %[[ROTATED:.*]] = tensor.extract %[[INPUT]]{{\[}}%[[I0]], %[[I1]], %[[I2]], %[[J]]] : tensor<1x3x2x6xf32>
// CHECK:             %[[SIGN_MUL:.*]] = arith.select %[[SIGN]], %[[ONE]], %[[MINUS_ONE]] : f32
// CHECK:             %[[OUT_I:.*]] = arith.mulf %[[IN]], %[[COS_EMB]] : f32
// CHECK:             %[[OUT_J:.*]] = arith.mulf %[[ROTATED]], %[[SIN_EMB]] : f32
// CHECK:             %[[SIGNED_J:.*]] = arith.mulf %[[OUT_J]], %[[SIGN_MUL]] : f32
// CHECK:             %[[OUT:.*]] = arith.addf %[[OUT_I]], %[[SIGNED_J]] : f32
// CHECK:             linalg.yield %[[OUT]] : f32
// CHECK:           tensor.cast %[[ROTARY]] : tensor<1x3x2x6xf32> to tensor<1x3x2x6xf32>
func.func @test_rotary_embedding(%arg0: !torch.vtensor<[1,3,2,6],f32>, %arg1: !torch.vtensor<[1,2],si64>, %arg2: !torch.vtensor<[4,3],f32>, %arg3: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[1,3,2,6],f32> {
  %int0 = torch.constant.int 0
  %float = torch.constant.float 1.000000e+00
  %0 = torch.onnx.rotary_embedding %arg0, %arg1, %arg2, %arg3, %int0, %int0, %int0, %int0, %float : !torch.vtensor<[1,3,2,6],f32>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[4,3],f32>, !torch.vtensor<[4,3],f32>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.float -> !torch.vtensor<[1,3,2,6],f32>
  return %0 : !torch.vtensor<[1,3,2,6],f32>
}

// -----

// A rank-3 input (batch, seq, num_heads * head_size) is expanded to
// (batch, seq, num_heads, head_size) with position_ids indexing dims 0 and 1.
// CHECK-DAG: #[[$ID:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$POS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CHECK-LABEL:   func.func @test_rotary_embedding_rank3(
// CHECK:           %[[INPUT:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[?,?,8],f32> -> tensor<?x?x8xf32>
// CHECK:           %[[BATCH:.*]] = tensor.dim %[[INPUT]], %{{.*}} : tensor<?x?x8xf32>
// CHECK:           %[[SEQ:.*]] = tensor.dim %[[INPUT]], %{{.*}} : tensor<?x?x8xf32>
// CHECK:           %[[EXPANDED:.*]] = tensor.expand_shape %[[INPUT]] {{\[}}[0], [1], [2, 3]] output_shape [%[[BATCH]], %[[SEQ]], 2, 4] : tensor<?x?x8xf32> into tensor<?x?x2x4xf32>
// CHECK:           %[[ROTARY:.*]] = linalg.generic {indexing_maps = [#[[$ID]], #[[$POS]], #[[$ID]]], {{.*}}} ins(%[[EXPANDED]], %{{.*}} : tensor<?x?x2x4xf32>, tensor<?x?xi64>)
// CHECK:             tensor.extract %[[EXPANDED]]
// CHECK:           %[[COLLAPSED:.*]] = tensor.collapse_shape %[[ROTARY]] {{\[}}[0], [1], [2, 3]] : tensor<?x?x2x4xf32> into tensor<?x?x8xf32>
// CHECK:           tensor.cast %[[COLLAPSED]]
func.func @test_rotary_embedding_rank3(%arg0: !torch.vtensor<[?,?,8],f32>, %arg1: !torch.vtensor<[?,?],si64>, %arg2: !torch.vtensor<[4,2],f32>, %arg3: !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[?,?,8],f32> {
  %int0 = torch.constant.int 0
  %float = torch.constant.float 1.000000e+00
  %0 = torch.onnx.rotary_embedding %arg0, %arg1, %arg2, %arg3, %int0, %int0, %int0, %int0, %float : !torch.vtensor<[?,?,8],f32>, !torch.vtensor<[?,?],si64>, !torch.vtensor<[4,2],f32>, !torch.vtensor<[4,2],f32>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.float -> !torch.vtensor<[?,?,8],f32>
  return %0 : !torch.vtensor<[?,?,8],f32>
}

// -----

// A scale other than 1 multiplies the rotated input before the collapse.
// CHECK-LABEL:   func.func @test_rotary_embedding_rank3_with_scale(
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<1x2x2x4xf32>
// CHECK:           %[[ROTARY:.*]] = linalg.generic
// CHECK:           %[[SCALE:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:           %[[SCALED:.*]] = linalg.generic {{.*}} ins(%[[ROTARY]] : tensor<1x2x2x4xf32>) outs(%[[EMPTY]] : tensor<1x2x2x4xf32>)
// CHECK:             %[[MUL:.*]] = arith.mulf %{{.*}}, %[[SCALE]] : f32
// CHECK:             linalg.yield %[[MUL]] : f32
// CHECK:           tensor.collapse_shape %[[SCALED]]
func.func @test_rotary_embedding_rank3_with_scale(%arg0: !torch.vtensor<[1,2,8],f32>, %arg1: !torch.vtensor<[1,2],si64>, %arg2: !torch.vtensor<[4,2],f32>, %arg3: !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[1,2,8],f32> {
  %int0 = torch.constant.int 0
  %float = torch.constant.float 2.000000e+00
  %0 = torch.onnx.rotary_embedding %arg0, %arg1, %arg2, %arg3, %int0, %int0, %int0, %int0, %float : !torch.vtensor<[1,2,8],f32>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[4,2],f32>, !torch.vtensor<[4,2],f32>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.float -> !torch.vtensor<[1,2,8],f32>
  return %0 : !torch.vtensor<[1,2,8],f32>
}

// -----

// A rank-3 hidden size that is not num_heads * head_size is rejected.
func.func @negative_test_rotary_embedding_rank3_hidden_size(%arg0: !torch.vtensor<[1,2,10],f32>, %arg1: !torch.vtensor<[1,2],si64>, %arg2: !torch.vtensor<[4,2],f32>, %arg3: !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[1,2,10],f32> {
  %int0 = torch.constant.int 0
  %float = torch.constant.float 1.000000e+00
  // expected-error @+1 {{failed to legalize operation 'torch.onnx.rotary_embedding'}}
  %0 = torch.onnx.rotary_embedding %arg0, %arg1, %arg2, %arg3, %int0, %int0, %int0, %int0, %float : !torch.vtensor<[1,2,10],f32>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[4,2],f32>, !torch.vtensor<[4,2],f32>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.float -> !torch.vtensor<[1,2,10],f32>
  return %0 : !torch.vtensor<[1,2,10],f32>
}

// -----

// An empty cos_cache is rejected.
func.func @negative_test_rotary_embedding_empty_cache(%arg0: !torch.vtensor<[1,2,8],f32>, %arg1: !torch.vtensor<[1,2],si64>, %arg2: !torch.vtensor<[4,0],f32>, %arg3: !torch.vtensor<[4,0],f32>) -> !torch.vtensor<[1,2,8],f32> {
  %int0 = torch.constant.int 0
  %float = torch.constant.float 1.000000e+00
  // expected-error @+1 {{failed to legalize operation 'torch.onnx.rotary_embedding'}}
  %0 = torch.onnx.rotary_embedding %arg0, %arg1, %arg2, %arg3, %int0, %int0, %int0, %int0, %float : !torch.vtensor<[1,2,8],f32>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[4,0],f32>, !torch.vtensor<[4,0],f32>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.float -> !torch.vtensor<[1,2,8],f32>
  return %0 : !torch.vtensor<[1,2,8],f32>
}

// -----

// position_ids that do not match the input's (batch, seq) are rejected.
func.func @negative_test_rotary_embedding_position_ids_shape(%arg0: !torch.vtensor<[1,2,8],f32>, %arg1: !torch.vtensor<[1,3],si64>, %arg2: !torch.vtensor<[4,2],f32>, %arg3: !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[1,2,8],f32> {
  %int0 = torch.constant.int 0
  %float = torch.constant.float 1.000000e+00
  // expected-error @+1 {{failed to legalize operation 'torch.onnx.rotary_embedding'}}
  %0 = torch.onnx.rotary_embedding %arg0, %arg1, %arg2, %arg3, %int0, %int0, %int0, %int0, %float : !torch.vtensor<[1,2,8],f32>, !torch.vtensor<[1,3],si64>, !torch.vtensor<[4,2],f32>, !torch.vtensor<[4,2],f32>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.float -> !torch.vtensor<[1,2,8],f32>
  return %0 : !torch.vtensor<[1,2,8],f32>
}

// -----

// A result shape that does not match the input is rejected.
func.func @negative_test_rotary_embedding_result_shape(%arg0: !torch.vtensor<[1,2,8],f32>, %arg1: !torch.vtensor<[1,2],si64>, %arg2: !torch.vtensor<[4,2],f32>, %arg3: !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[1,2,6],f32> {
  %int0 = torch.constant.int 0
  %float = torch.constant.float 1.000000e+00
  // expected-error @+1 {{failed to legalize operation 'torch.onnx.rotary_embedding'}}
  %0 = torch.onnx.rotary_embedding %arg0, %arg1, %arg2, %arg3, %int0, %int0, %int0, %int0, %float : !torch.vtensor<[1,2,8],f32>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[4,2],f32>, !torch.vtensor<[4,2],f32>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.float -> !torch.vtensor<[1,2,6],f32>
  return %0 : !torch.vtensor<[1,2,6],f32>
}

// -----

// CHECK-LABEL: func.func @torch.ops.aten.replication_pad3d$basic(
// CHECK-SAME: %[[ARG_0:.*]]: !torch.vtensor<[4,3,5],f32>) -> !torch.vtensor<[7,7,6],f32>
// CHECK: %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[4,3,5],f32> -> tensor<4x3x5xf32>
// CHECK-DAG: %[[INT0:.*]] = torch.constant.int 0
// CHECK-DAG: %[[INT1:.*]] = torch.constant.int 1
// CHECK-DAG: %[[INT3:.*]] = torch.constant.int 3
// CHECK: %[[IDX5:.*]] = arith.constant 5 : index
// CHECK: %[[IDX1:.*]] = arith.constant 1 : index
// CHECK: %[[SUB2:.*]] = arith.subi %[[IDX5]], %[[IDX1]] : index
// CHECK: %[[SLICE1:.*]] = tensor.extract_slice %[[T0]][0, 0, %[[SUB2]]] [4, 3, 1] [1, 1, 1] : tensor<4x3x5xf32> to tensor<4x3x1xf32>
// CHECK: %[[CONCAT1:.*]] = tensor.concat dim(2) %[[T0]], %[[SLICE1]] : (tensor<4x3x5xf32>, tensor<4x3x1xf32>) -> tensor<4x3x6xf32>
// CHECK: %[[SLICE2:.*]] = tensor.extract_slice %[[CONCAT1]][0, 0, 0] [4, 1, 6] [1, 1, 1] : tensor<4x3x6xf32> to tensor<4x1x6xf32>
// CHECK: %[[CONCAT2:.*]] = tensor.concat dim(1) %[[SLICE2]], %[[SLICE2]], %[[SLICE2]] : (tensor<4x1x6xf32>, tensor<4x1x6xf32>, tensor<4x1x6xf32>) -> tensor<4x3x6xf32>
// CHECK: %[[SUB3:.*]] = arith.subi {{.*}}, {{.*}} : index
// CHECK: %[[SLICE3:.*]] = tensor.extract_slice %[[CONCAT1]][0, %[[SUB3]], 0] [4, 1, 6] [1, 1, 1] : tensor<4x3x6xf32> to tensor<4x1x6xf32>
// CHECK: %[[CONCAT3:.*]] = tensor.concat dim(1) %[[CONCAT2]], %[[CONCAT1]], %[[SLICE3]] : (tensor<4x3x6xf32>, tensor<4x3x6xf32>, tensor<4x1x6xf32>) -> tensor<4x7x6xf32>
// CHECK: %[[SUB4:.*]] = arith.subi {{.*}}, {{.*}} : index
// CHECK: %[[SLICE4:.*]] = tensor.extract_slice %[[CONCAT3]][%[[SUB4]], 0, 0] [1, 7, 6] [1, 1, 1] : tensor<4x7x6xf32> to tensor<1x7x6xf32>
// CHECK: %[[CONCAT4:.*]] = tensor.concat dim(0) %[[SLICE4]], %[[SLICE4]], %[[SLICE4]] : (tensor<1x7x6xf32>, tensor<1x7x6xf32>, tensor<1x7x6xf32>) -> tensor<3x7x6xf32>
// CHECK: %[[CONCAT5:.*]] = tensor.concat dim(0) %[[CONCAT3]], %[[CONCAT4]] : (tensor<4x7x6xf32>, tensor<3x7x6xf32>) -> tensor<7x7x6xf32>
// CHECK: %[[CAST:.*]] = tensor.cast %[[CONCAT5]] : tensor<7x7x6xf32> to tensor<7x7x6xf32>
// CHECK: %[[OUT:.*]] = torch_c.from_builtin_tensor %[[CAST]] : tensor<7x7x6xf32> -> !torch.vtensor<[7,7,6],f32>
// CHECK: return %[[OUT]] : !torch.vtensor<[7,7,6],f32>
func.func @torch.ops.aten.replication_pad3d$basic(%arg0: !torch.vtensor<[4,3,5],f32>) -> !torch.vtensor<[7,7,6],f32> {
  %c0 = torch.constant.int 0
  %c1 = torch.constant.int 1
  %c3 = torch.constant.int 3
  %padding = torch.prim.ListConstruct %c0, %c1, %c3, %c1, %c0, %c3 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.replication_pad3d %arg0, %padding : !torch.vtensor<[4,3,5],f32>, !torch.list<int> -> !torch.vtensor<[7,7,6],f32>
  return %0 : !torch.vtensor<[7,7,6],f32>
}

// -----

// CHECK-LABEL: func.func @torch.ops.aten.replication_pad2d$basic
// CHECK: [[TENSOR:%.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[4,3,5],f32> -> tensor<4x3x5xf32>
// CHECK: [[INT1:%.*]] = torch.constant.int 1
// CHECK: [[INT3:%.*]] = torch.constant.int 3
// CHECK: [[PAD_LIST:%.*]] = torch.prim.ListConstruct [[INT1]], [[INT3]], [[INT1]], [[INT3]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK: [[SLICE_LEFT:%.*]] = tensor.extract_slice [[TENSOR]][0, 0, 0] [4, 3, 1] [1, 1, 1] : tensor<4x3x5xf32> to tensor<4x3x1xf32>
// CHECK: [[IDX_RIGHT:%.*]] = arith.subi %c5_5, %c1_6 : index
// CHECK: [[SLICE_RIGHT:%.*]] = tensor.extract_slice [[TENSOR]][0, 0, [[IDX_RIGHT]]] [4, 3, 1] [1, 1, 1] : tensor<4x3x5xf32> to tensor<4x3x1xf32>
// CHECK: [[RIGHT_PAD:%.*]] = tensor.concat dim(2) [[SLICE_RIGHT]], [[SLICE_RIGHT]], [[SLICE_RIGHT]] : (tensor<4x3x1xf32>, tensor<4x3x1xf32>, tensor<4x3x1xf32>) -> tensor<4x3x3xf32>
// CHECK: [[PADDED_W:%.*]] = tensor.concat dim(2) [[SLICE_LEFT]], [[TENSOR]], [[RIGHT_PAD]] : (tensor<4x3x1xf32>, tensor<4x3x5xf32>, tensor<4x3x3xf32>) -> tensor<4x3x9xf32>
// CHECK: [[SLICE_TOP:%.*]] = tensor.extract_slice [[PADDED_W]][0, 0, 0] [4, 1, 9] [1, 1, 1] : tensor<4x3x9xf32> to tensor<4x1x9xf32>
// CHECK: [[IDX_BOTTOM:%.*]] = arith.subi %c3_18, %c1_21 : index
// CHECK: [[SLICE_BOTTOM:%.*]] = tensor.extract_slice [[PADDED_W]][0, [[IDX_BOTTOM]], 0] [4, 1, 9] [1, 1, 1] : tensor<4x3x9xf32> to tensor<4x1x9xf32>
// CHECK: [[BOTTOM_PAD:%.*]] = tensor.concat dim(1) [[SLICE_BOTTOM]], [[SLICE_BOTTOM]], [[SLICE_BOTTOM]] : (tensor<4x1x9xf32>, tensor<4x1x9xf32>, tensor<4x1x9xf32>) -> tensor<4x3x9xf32>
// CHECK: [[PADDED_HW:%.*]] = tensor.concat dim(1) [[SLICE_TOP]], [[PADDED_W]], [[BOTTOM_PAD]] : (tensor<4x1x9xf32>, tensor<4x3x9xf32>, tensor<4x3x9xf32>) -> tensor<4x7x9xf32>
// CHECK: [[CAST:%.*]] = tensor.cast [[PADDED_HW]] : tensor<4x7x9xf32> to tensor<4x7x9xf32>
// CHECK: [[TORCH_TENSOR:%.*]] = torch_c.from_builtin_tensor [[CAST]] : tensor<4x7x9xf32> -> !torch.vtensor<[4,7,9],f32>
// CHECK: return [[TORCH_TENSOR]] : !torch.vtensor<[4,7,9],f32>
func.func @torch.ops.aten.replication_pad2d$basic(%arg0: !torch.vtensor<[4,3,5],f32>) -> !torch.vtensor<[4,7,9],f32> {
  %c1 = torch.constant.int 1
  %c3 = torch.constant.int 3
  %padding = torch.prim.ListConstruct %c1, %c3, %c1, %c3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.replication_pad2d %arg0, %padding : !torch.vtensor<[4,3,5],f32>, !torch.list<int> -> !torch.vtensor<[4,7,9],f32>
  return %0 : !torch.vtensor<[4,7,9],f32>
}

// -----


// CHECK-LABEL: func.func @torch.ops.aten.replication_pad1d$basic
// CHECK: [[TENSOR:%.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[3,5],f32> -> tensor<3x5xf32>
// CHECK: [[INT1:%.*]] = torch.constant.int 1
// CHECK: [[INT2:%.*]] = torch.constant.int 2
// CHECK: [[PAD_LIST:%.*]] = torch.prim.ListConstruct [[INT1]], [[INT2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: [[C0:%.*]] = arith.constant 0 : index
// CHECK: [[C3:%.*]] = arith.constant 3 : index
// CHECK: [[C1:%.*]] = arith.constant 1 : index
// CHECK: [[C5:%.*]] = arith.constant 5 : index
// CHECK: [[SLICE_LEFT:%.*]] = tensor.extract_slice [[TENSOR]][0, 0] [3, 1] [1, 1] : tensor<3x5xf32> to tensor<3x1xf32>
// CHECK: [[C0_0:%.*]] = arith.constant 0 : index
// CHECK: [[C3_1:%.*]] = arith.constant 3 : index
// CHECK: [[C1_2:%.*]] = arith.constant 1 : index
// CHECK: [[C5_3:%.*]] = arith.constant 5 : index
// CHECK: [[C1_4:%.*]] = arith.constant 1 : index
// CHECK: [[IDX_RIGHT:%.*]] = arith.subi [[C5_3]], [[C1_4]] : index
// CHECK: [[SLICE_RIGHT:%.*]] = tensor.extract_slice [[TENSOR]][0, [[IDX_RIGHT]]] [3, 1] [1, 1] : tensor<3x5xf32> to tensor<3x1xf32>
// CHECK: [[RIGHT_PAD:%.*]] = tensor.concat dim(1) [[SLICE_RIGHT]], [[SLICE_RIGHT]] : (tensor<3x1xf32>, tensor<3x1xf32>) -> tensor<3x2xf32>
// CHECK: [[PADDED:%.*]] = tensor.concat dim(1) [[SLICE_LEFT]], [[TENSOR]], [[RIGHT_PAD]] : (tensor<3x1xf32>, tensor<3x5xf32>, tensor<3x2xf32>) -> tensor<3x8xf32>
// CHECK: [[CAST:%.*]] = tensor.cast [[PADDED]] : tensor<3x8xf32> to tensor<3x8xf32>
// CHECK: [[TORCH_TENSOR:%.*]] = torch_c.from_builtin_tensor [[CAST]] : tensor<3x8xf32> -> !torch.vtensor<[3,8],f32>
// CHECK: return [[TORCH_TENSOR]] : !torch.vtensor<[3,8],f32>
func.func @torch.ops.aten.replication_pad1d$basic(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[3,8],f32> {
  %c1 = torch.constant.int 1
  %c2 = torch.constant.int 2
  %padding = torch.prim.ListConstruct %c1, %c2 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.replication_pad1d %arg0, %padding : !torch.vtensor<[3,5],f32>, !torch.list<int> -> !torch.vtensor<[3,8],f32>
  return %0 : !torch.vtensor<[3,8],f32>
}

// -----
// COM: this test only locks down that allow-non-finites config produces inf/realmax correctly
// CHECK-LABEL: func.func @torch.aten.min.dim$basic
// CHECK: arith.constant 0x7F800000 : f32
// CHECK-NOT: arith.constant 3.40282347E+38 : f32
// UNSUPPORTED-NON-FINITES: arith.constant 3.40282347E+38 : f32
// UNSUPPORTED-NON-FINITES-NOT: arith.constant 0x7F800000 : f32
func.func @torch.aten.min.dim$basic(%arg0: tensor<3x2x3xf32>) -> tensor<3x2x1xf32> {
  %0 = torch_c.from_builtin_tensor %arg0 : tensor<3x2x3xf32> -> !torch.vtensor<[3,2,3],f32>
  %true = torch.constant.bool true
  %int2 = torch.constant.int 2
  %values, %indices = torch.aten.min.dim %0, %int2, %true : !torch.vtensor<[3,2,3],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1],f32>, !torch.vtensor<[3,2,1],si64>
  %1 = torch_c.to_builtin_tensor %values : !torch.vtensor<[3,2,1],f32> -> tensor<3x2x1xf32>
  return %1 : tensor<3x2x1xf32>
}

// -----
// COM: this test only locks down that allow-non-finites config produces inf/realmax correctly
// CHECK-LABEL: func.func @torch.aten.max.dim$basic
// CHECK: arith.constant 0xFF800000 : f32
// CHECK-NOT: arith.constant -3.40282347E+38 : f32
// UNSUPPORTED-NON-FINITES: arith.constant -3.40282347E+38 : f32
// UNSUPPORTED-NON-FINITES-NOT: arith.constant 0xFF800000 : f32
func.func @torch.aten.max.dim$basic(%arg0: tensor<3x2x3xf32>) -> tensor<3x2x1xf32> {
  %0 = torch_c.from_builtin_tensor %arg0 : tensor<3x2x3xf32> -> !torch.vtensor<[3,2,3],f32>
  %true = torch.constant.bool true
  %int2 = torch.constant.int 2
  %values, %indices = torch.aten.max.dim %0, %int2, %true : !torch.vtensor<[3,2,3],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1],f32>, !torch.vtensor<[3,2,1],si64>
  %1 = torch_c.to_builtin_tensor %values : !torch.vtensor<[3,2,1],f32> -> tensor<3x2x1xf32>
  return %1 : tensor<3x2x1xf32>
}


// -----


// CHECK-DAG: #[[$MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
// CHECK-LABEL: func.func @torch.ops.aten.anydim$basic(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<1x16x26x26xi1>) -> !torch.vtensor<[1,16,26,1],i1>
// CHECK: %[[INT:.*]] = torch.constant.int -1
// CHECK: %[[TRUE:.*]] = torch.constant.bool true
// CHECK: %[[T0:.*]] = torch_c.from_builtin_tensor %[[ARG0]] : tensor<1x16x26x26xi1> -> !torch.vtensor<[1,16,26,26],i1>
// CHECK: %[[T1:.*]] = torch_c.to_builtin_tensor %[[T0]] : !torch.vtensor<[1,16,26,26],i1> -> tensor<1x16x26x26xi1>
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM0:.*]] = tensor.dim %[[T1]], %[[C0]] : tensor<1x16x26x26xi1>
// CHECK: %[[C1_0:.*]] = arith.constant 1 : index
// CHECK: %[[DIM1:.*]] = tensor.dim %[[T1]], %[[C1_0]] : tensor<1x16x26x26xi1>
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[DIM2:.*]] = tensor.dim %[[T1]], %[[C2]] : tensor<1x16x26x26xi1>
// CHECK: %[[C3:.*]] = arith.constant 3 : index
// CHECK: %[[DIM3:.*]] = tensor.dim %[[T1]], %[[C3]] : tensor<1x16x26x26xi1>
// CHECK: %[[EMPTY:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]], %[[DIM2]]) : tensor<?x?x?x1xi1>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[FALSE]] : i1) outs(%[[EMPTY]] : tensor<?x?x?x1xi1>) -> tensor<?x?x?x1xi1>
// CHECK: %[[GENERIC:.*]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%[[T1]] : tensor<1x16x26x26xi1>) outs(%[[FILL]] : tensor<?x?x?x1xi1>) {
// CHECK: ^bb0(%[[IN:.*]]: i1, %[[OUT:.*]]: i1):
// CHECK:   %[[ORI:.*]] = arith.ori %[[IN]], %[[OUT]] : i1
// CHECK:   linalg.yield %[[ORI]] : i1
// CHECK: } -> tensor<?x?x?x1xi1>
// CHECK: %[[CAST:.*]] = tensor.cast %[[GENERIC]] : tensor<?x?x?x1xi1> to tensor<1x16x26x1xi1>
// CHECK: %[[RET:.*]] = torch_c.from_builtin_tensor %[[CAST]] : tensor<1x16x26x1xi1> -> !torch.vtensor<[1,16,26,1],i1>
// CHECK: return %[[RET]] : !torch.vtensor<[1,16,26,1],i1>
func.func @torch.ops.aten.anydim$basic(%arg0: tensor<1x16x26x26xi1>) -> !torch.vtensor<[1,16,26,1],i1> {
  %int-1 = torch.constant.int -1
  %true = torch.constant.bool true
  %0 = torch_c.from_builtin_tensor %arg0 : tensor<1x16x26x26xi1> -> !torch.vtensor<[1,16,26,26],i1>
  %1 = torch.aten.any.dim %0, %int-1, %true : !torch.vtensor<[1,16,26,26],i1>, !torch.int, !torch.bool -> !torch.vtensor<[1,16,26,1],i1>
  return %1 : !torch.vtensor<[1,16,26,1],i1>
}

// -----

// Per PyTorch docs, torch.cat allows "a 1-D empty tensor with size (0,)"
// alongside operands of any rank. The linalg lowering must skip these.
// CHECK-LABEL: func.func @torch.aten.cat$rank1_empty
//  CHECK-SAME:   %[[ARG0:.*]]: !torch.vtensor<[1,8,?,128],f16>, %[[ARG1:.*]]: !torch.vtensor<[0],f16>
//       CHECK:   %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1,8,?,128],f16> -> tensor<1x8x?x128xf16>
//       CHECK:   %[[CONCAT:.*]] = tensor.concat dim(2) %[[T0]] : (tensor<1x8x?x128xf16>) -> tensor<1x8x?x128xf16>
//       CHECK:   %[[T1:.*]] = torch_c.from_builtin_tensor %[[CONCAT]] : tensor<1x8x?x128xf16> -> !torch.vtensor<[1,8,?,128],f16>
//       CHECK:   return %[[T1]]
func.func @torch.aten.cat$rank1_empty(%arg0: !torch.vtensor<[1,8,?,128],f16>, %arg1: !torch.vtensor<[0],f16>) -> !torch.vtensor<[1,8,?,128],f16> {
  %int-2 = torch.constant.int -2
  %0 = torch.prim.ListConstruct %arg1, %arg0 : (!torch.vtensor<[0],f16>, !torch.vtensor<[1,8,?,128],f16>) -> !torch.list<vtensor>
  %1 = torch.aten.cat %0, %int-2 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,8,?,128],f16>
  return %1 : !torch.vtensor<[1,8,?,128],f16>
}

// -----
// CHECK-LABEL: func.func @torch.aten.max$bool
// CHECK:         %[[INIT:.+]] = arith.constant false
// CHECK:         linalg.fill ins(%[[INIT]] : i1)
// CHECK:         linalg.generic
// CHECK:         ^bb0(%[[IN:.+]]: i1, %[[OUT:.+]]: i1):
// CHECK-NEXT:      %[[RES:.+]] = arith.maxui %[[IN]], %[[OUT]] : i1
func.func @torch.aten.max$bool(%arg0: !torch.vtensor<[4],i1>) -> !torch.vtensor<[],i1> {
  %0 = torch.aten.max %arg0 : !torch.vtensor<[4],i1> -> !torch.vtensor<[],i1>
  return %0 : !torch.vtensor<[],i1>
}

// -----
// CHECK-LABEL: func.func @torch.aten.min$bool
// CHECK:         %[[INIT:.+]] = arith.constant true
// CHECK:         linalg.fill ins(%[[INIT]] : i1)
// CHECK:         linalg.generic
// CHECK:         ^bb0(%[[IN:.+]]: i1, %[[OUT:.+]]: i1):
// CHECK-NEXT:      %[[RES:.+]] = arith.minui %[[IN]], %[[OUT]] : i1
func.func @torch.aten.min$bool(%arg0: !torch.vtensor<[4],i1>) -> !torch.vtensor<[],i1> {
  %0 = torch.aten.min %arg0 : !torch.vtensor<[4],i1> -> !torch.vtensor<[],i1>
  return %0 : !torch.vtensor<[],i1>
}

// -----
// CHECK-LABEL: func.func @torch.aten.max.dim$bool
// CHECK:         %[[INIT:.+]] = arith.constant false
// CHECK:         linalg.fill ins(%[[INIT]] : i1)
// CHECK:         linalg.generic
// CHECK:         ^bb0(%[[IN:.+]]: i1, {{.*}}: i1, {{.*}}: i64):
// CHECK:           arith.maxui %[[IN]], {{.*}} : i1
// CHECK:           arith.cmpi ugt, {{.*}} : i1
func.func @torch.aten.max.dim$bool(%arg0: !torch.vtensor<[3,4],i1>) -> !torch.vtensor<[3],i1> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %values, %indices = torch.aten.max.dim %arg0, %int1, %false : !torch.vtensor<[3,4],i1>, !torch.int, !torch.bool -> !torch.vtensor<[3],i1>, !torch.vtensor<[3],si64>
  return %values : !torch.vtensor<[3],i1>
}

// -----
// CHECK-LABEL: func.func @torch.aten.min.dim$bool
// CHECK:         %[[INIT:.+]] = arith.constant true
// CHECK:         linalg.fill ins(%[[INIT]] : i1)
// CHECK:         linalg.generic
// CHECK:         ^bb0(%[[IN:.+]]: i1, {{.*}}: i1, {{.*}}: i64):
// CHECK:           arith.minui %[[IN]], {{.*}} : i1
// CHECK:           arith.cmpi ult, {{.*}} : i1
func.func @torch.aten.min.dim$bool(%arg0: !torch.vtensor<[3,4],i1>) -> !torch.vtensor<[3],i1> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %values, %indices = torch.aten.min.dim %arg0, %int1, %false : !torch.vtensor<[3,4],i1>, !torch.int, !torch.bool -> !torch.vtensor<[3],i1>, !torch.vtensor<[3],si64>
  return %values : !torch.vtensor<[3],i1>
}
