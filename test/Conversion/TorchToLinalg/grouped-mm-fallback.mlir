// RUN: torch-mlir-opt %s -pass-pipeline='builtin.module(func.func(torch-match-quantized-custom-ops),torchdynamo-export-to-torch-backend-pipeline{extra-library=},torch-backend-to-linalg-on-tensors-backend-pipeline)' | FileCheck %s

// CHECK-LABEL: func.func @forward
// CHECK-NOT: torch.operator
// CHECK-NOT: torch.aten._grouped_mm
// CHECK: scf.for
// CHECK: tensor.extract_slice
// CHECK: linalg.matmul
// CHECK: tensor.insert_slice
func.func @forward(%input: !torch.vtensor<[4,1024],f32>, %weight: !torch.vtensor<[16,1024,512],f32>, %offsets: !torch.vtensor<[16],si64>) -> !torch.vtensor<[4,512],f32> {
  %0 = torch.operator "torch.transformers.grouped_mm_fallback"(%input, %weight, %offsets) : (!torch.vtensor<[4,1024],f32>, !torch.vtensor<[16,1024,512],f32>, !torch.vtensor<[16],si64>) -> !torch.vtensor<[4,512],f32>
  return %0 : !torch.vtensor<[4,512],f32>
}
