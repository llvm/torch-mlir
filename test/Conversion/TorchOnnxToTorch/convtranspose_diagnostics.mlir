// RUN: torch-mlir-opt <%s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch

func.func @test_convtranspose_no_spatial_axis(
    %arg0: !torch.vtensor<[3],f32>,
    %arg1: !torch.vtensor<[3],f32>) -> !torch.vtensor<[5],f32>
    attributes {torch.onnx_meta.ir_version = 10 : si64,
                torch.onnx_meta.opset_version = 22 : si64,
                torch.onnx_meta.producer_name = "backend-test",
                torch.onnx_meta.producer_version = ""} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.ConvTranspose"(%arg0, %arg1)
      : (!torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[5],f32>
  return %0 : !torch.vtensor<[5],f32>
}

// -----

func.func @test_convtranspose_result_rank_mismatch(
    %arg0: !torch.vtensor<[1,1,3,3],f32>,
    %arg1: !torch.vtensor<[1,2,3,3],f32>) -> !torch.vtensor<[6],f32>
    attributes {torch.onnx_meta.ir_version = 10 : si64,
                torch.onnx_meta.opset_version = 22 : si64,
                torch.onnx_meta.producer_name = "backend-test",
                torch.onnx_meta.producer_version = ""} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.ConvTranspose"(%arg0, %arg1)
      : (!torch.vtensor<[1,1,3,3],f32>, !torch.vtensor<[1,2,3,3],f32>) -> !torch.vtensor<[6],f32>
  return %0 : !torch.vtensor<[6],f32>
}

// -----

func.func @test_convtranspose_output_shape_dynamic_spatial_dims(
    %arg0: !torch.vtensor<[1,1,?,?],f32>,
    %arg1: !torch.vtensor<[1,2,4,4],f32>) -> !torch.vtensor<[1,2,8,8],f32>
    attributes {torch.onnx_meta.ir_version = 10 : si64,
                torch.onnx_meta.opset_version = 22 : si64,
                torch.onnx_meta.producer_name = "backend-test",
                torch.onnx_meta.producer_version = ""} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.ConvTranspose"(%arg0, %arg1) {
    torch.onnx.auto_pad = "SAME_UPPER",
    torch.onnx.output_shape = [8 : si64, 8 : si64],
    torch.onnx.strides = [2 : si64, 2 : si64]
  } : (!torch.vtensor<[1,1,?,?],f32>, !torch.vtensor<[1,2,4,4],f32>) -> !torch.vtensor<[1,2,8,8],f32>
  return %0 : !torch.vtensor<[1,2,8,8],f32>
}

// -----

func.func @test_convtranspose_autopad_same_dynamic_kernel(
    %arg0: !torch.vtensor<[1,1,3,3],f32>,
    %arg1: !torch.vtensor<[1,2,?,?],f32>) -> !torch.vtensor<[1,2,3,3],f32>
    attributes {torch.onnx_meta.ir_version = 10 : si64,
                torch.onnx_meta.opset_version = 22 : si64,
                torch.onnx_meta.producer_name = "backend-test",
                torch.onnx_meta.producer_version = ""} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.ConvTranspose"(%arg0, %arg1) {
    torch.onnx.auto_pad = "SAME_UPPER"
  } : (!torch.vtensor<[1,1,3,3],f32>, !torch.vtensor<[1,2,?,?],f32>) -> !torch.vtensor<[1,2,3,3],f32>
  return %0 : !torch.vtensor<[1,2,3,3],f32>
}
