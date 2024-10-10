// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @torch.aten.squeeze.dim$dynamic
func.func @torch.aten.squeeze.dim$dynamic(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "tf2onnx", torch.onnx_meta.producer_version = "1.5.2"} {
  // CHECK: %[[BUILTIN_TENSOR:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
  // CHECK: %[[C0:.*]] = torch.constant.int 0
  // CHECK: %[[C0_1:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM:.*]] = tensor.dim %[[BUILTIN_TENSOR]], %[[C0_1]] : tensor<?x?x?xf32>
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[CMPI:.*]] = arith.cmpi eq, %[[DIM]], %[[C1]] : index
  // CHECK: cf.assert %[[CMPI]], "Expected dynamic squeeze dim size to be statically 1"
  // CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[BUILTIN_TENSOR]] {{\[\[}}0, 1], [2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
  // CHECK: %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[COLLAPSED]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
  %int0 = torch.constant.int 0
  %1 = torch.aten.squeeze.dim %arg0, %int0 : !torch.vtensor<[?,?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %1 : !torch.vtensor<[?,?],f32>
}
