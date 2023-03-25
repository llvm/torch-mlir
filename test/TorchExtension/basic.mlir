// RUN: torch-mlir-opt %s -test-torch-dialect-extension -split-input-file | FileCheck %s

module attributes {torch.debug_module_name = "CustomOpExampleModule"} {
  func.func @forward(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
    %int2 = torch.constant.int 2
    %0 = torch.aten.mul.Scalar %arg0, %int2 : !torch.vtensor<[3,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>
    // CHECK: torch.goofy.identity %{{.*}} {bob = "uncle"}
    %1 = torch.goofy.identity %0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
    return %1 : !torch.vtensor<[3,4],f32>
  }
}