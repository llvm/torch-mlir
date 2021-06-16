// RUN: npcomp-opt <%s -convert-torch-to-std | FileCheck %s


// CHECK-LABEL:   func @torch.aten.dim(
// CHECK-SAME:                         %[[ARG:.*]]: !torch.vtensor<*,f32>) -> !torch.int {
// CHECK:           %[[BUILTIN_TENSOR:.*]] = torch.to_builtin_tensor %[[ARG]] : !torch.vtensor<*,f32> -> tensor<*xf32>
// CHECK:           %[[RANK:.*]] = rank %[[BUILTIN_TENSOR]] : tensor<*xf32>
// CHECK:           %[[RANK_I64:.*]] = index_cast %[[RANK]] : index to i64
// CHECK:           %[[RANK_TORCH_INT:.*]] = torch.from_i64 %[[RANK_I64]]
// CHECK:           return %[[RANK_TORCH_INT]] : !torch.int
func @torch.aten.dim(%arg0: !torch.vtensor<*,f32>) -> !torch.int {
  %0 = torch.aten.dim %arg0 : !torch.vtensor<*,f32> -> !torch.int
  return %0 : !torch.int
}

// CHECK-LABEL:   func @torch.aten.ne.int(
// CHECK-SAME:                            %[[LHS:.*]]: !torch.int,
// CHECK-SAME:                            %[[RHS:.*]]: !torch.int) -> !torch.bool {
// CHECK:           %[[LHS_I64:.*]] = torch.to_i64 %[[LHS]]
// CHECK:           %[[RHS_I64:.*]] = torch.to_i64 %[[RHS]]
// CHECK:           %[[CMP:.*]] = cmpi ne, %[[LHS_I64]], %[[RHS_I64]] : i64
// CHECK:           %[[CMP_TORCH_BOOL:.*]] = torch.from_i1 %[[CMP]]
// CHECK:           return %[[CMP_TORCH_BOOL]] : !torch.bool
func @torch.aten.ne.int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.bool {
  %0 = torch.aten.ne.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func @torch.aten.gt.int(
// CHECK-SAME:                            %[[LHS:.*]]: !torch.int,
// CHECK-SAME:                            %[[RHS:.*]]: !torch.int) -> !torch.bool {
// CHECK:           %[[LHS_I64:.*]] = torch.to_i64 %[[LHS]]
// CHECK:           %[[RHS_I64:.*]] = torch.to_i64 %[[RHS]]
// CHECK:           %[[CMP:.*]] = cmpi sgt, %[[LHS_I64]], %[[RHS_I64]] : i64
// CHECK:           %[[CMP_TORCH_BOOL:.*]] = torch.from_i1 %[[CMP]]
// CHECK:           return %[[CMP_TORCH_BOOL]] : !torch.bool
func @torch.aten.gt.int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.bool {
  %0 = torch.aten.gt.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.bool
  return %0 : !torch.bool
}

// CHECK-LABEL:   func @torch.tensor$value() -> !torch.vtensor<[],f32> {
// CHECK:           %[[CST:.*]] = constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VTENSOR:.*]] = torch.from_builtin_tensor %[[CST]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:           return %[[VTENSOR]] : !torch.vtensor<[],f32>
func @torch.tensor$value() -> !torch.vtensor<[],f32> {
  %0 = torch.tensor(dense<0.0> : tensor<f32>) : !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// CHECK-LABEL:   func @torch.tensor$nonval() -> !torch.tensor<[],f32> {
// CHECK:           %[[CST:.*]] = constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VTENSOR:.*]] = torch.from_builtin_tensor %[[CST]] : tensor<f32> -> !torch.vtensor<[],f32>
// CHECK:           %[[NONVAL:.*]] = torch.copy.tensor %[[VTENSOR]] : !torch.vtensor<[],f32> -> !torch.tensor<[],f32>
// CHECK:           return %[[NONVAL]] : !torch.tensor<[],f32>
func @torch.tensor$nonval() -> !torch.tensor<[],f32> {
  %0 = torch.tensor(dense<0.0> : tensor<f32>) : !torch.tensor<[],f32>
  return %0 : !torch.tensor<[],f32>
}
