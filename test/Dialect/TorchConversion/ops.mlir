// RUN: npcomp-opt %s | npcomp-opt | FileCheck %s

// CHECK-LABEL: func @builtin_tensor_interop(
func @builtin_tensor_interop(%arg0: tensor<*xf32>, %arg1: tensor<3x?xsi8>, %arg2: !torch.vtensor<*,f32>, %arg3: !torch.vtensor<[3,?],si8>) {
  // CHECK: torch_c.from_builtin_tensor %arg0 : tensor<*xf32> -> !torch.vtensor<*,f32>
  %0 = torch_c.from_builtin_tensor %arg0 : tensor<*xf32> -> !torch.vtensor<*,f32>
  // CHECK: torch_c.from_builtin_tensor %arg1 : tensor<3x?xsi8> -> !torch.vtensor<[3,?],si8>
  %1 = torch_c.from_builtin_tensor %arg1 : tensor<3x?xsi8> -> !torch.vtensor<[3,?],si8>
  // CHECK: torch_c.to_builtin_tensor %arg2 : !torch.vtensor<*,f32> -> tensor<*xf32>
  %2 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<*,f32> -> tensor<*xf32>
  // CHECK: torch_c.to_builtin_tensor %arg3 : !torch.vtensor<[3,?],si8> -> tensor<3x?xsi8>
  %3 = torch_c.to_builtin_tensor %arg3 : !torch.vtensor<[3,?],si8> -> tensor<3x?xsi8>
  return
}
