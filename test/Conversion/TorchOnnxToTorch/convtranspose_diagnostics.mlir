// RUN: torch-mlir-opt <%s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch

func.func @test_convtranspose_output_shape_with_conflicting_output_padding(
    %arg0: !torch.vtensor<[1,1,3,3],f32>,
    %arg1: !torch.vtensor<[1,2,3,3],f32>) -> !torch.vtensor<[1,2,10,8],f32>
    attributes {torch.onnx_meta.ir_version = 10 : si64,
                torch.onnx_meta.opset_version = 22 : si64,
                torch.onnx_meta.producer_name = "backend-test",
                torch.onnx_meta.producer_version = ""} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.ConvTranspose"(%arg0, %arg1) {
    torch.onnx.output_padding = [0 : si64, 1 : si64],
    torch.onnx.output_shape = [10 : si64, 8 : si64],
    torch.onnx.strides = [3 : si64, 2 : si64]
  } : (!torch.vtensor<[1,1,3,3],f32>, !torch.vtensor<[1,2,3,3],f32>) -> !torch.vtensor<[1,2,10,8],f32>
  return %0 : !torch.vtensor<[1,2,10,8],f32>
}
