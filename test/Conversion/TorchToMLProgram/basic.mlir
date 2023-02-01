// RUN: torch-mlir-opt <%s -convert-torch-to-mlprogram | FileCheck %s


// CHECK: #extern = #ml_program.extern : tensor<2x3xf32>
// CHECK: ml_program.global public @fc.weight(#extern) : tensor<2x3xf32>

// CHECK-LABEL:   func.func @torch.vtensor.external.literal() -> !torch.vtensor<[2,3],f32> {
// CHECK:           %[[BUILTIN_TENSOR:.*]] = ml_program.global_load_const @fc.weight : tensor<2x3xf32>
// CHECK:           %[[TORCH_TENSOR:.*]] = torch_c.from_builtin_tensor %[[BUILTIN_TENSOR]] : tensor<2x3xf32> -> !torch.vtensor<[2,3],f32>
// CHECK:           return %[[TORCH_TENSOR]] : !torch.vtensor<[2,3],f32>
func.func @torch.vtensor.external.literal() -> !torch.vtensor<[2,3],f32> {
  %0 = torch.vtensor.external.literal(@fc.weight) : !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}
