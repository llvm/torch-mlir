// RUN: torch-mlir-opt %s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch

func.func @test_convinteger_invalid_padding_size(
    %arg0: !torch.vtensor<[1,1,3,3],ui8>,
    %arg1: !torch.vtensor<[1,1,2,2],ui8>)
    -> !torch.vtensor<[1,1,2,2],si32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.ConvInteger"(%arg0, %arg1) {
    torch.onnx.pads = [0 : si64, 0 : si64, 0 : si64]
  } : (!torch.vtensor<[1,1,3,3],ui8>,
       !torch.vtensor<[1,1,2,2],ui8>)
       -> !torch.vtensor<[1,1,2,2],si32>
  return %0 : !torch.vtensor<[1,1,2,2],si32>
}

// -----

// per-channel w_zero_point length does not match the output channels.
func.func @test_convinteger_per_channel_weight_zp_length_mismatch(
    %arg0: !torch.vtensor<[1,2,3,3],ui8>,
    %arg1: !torch.vtensor<[3,2,2,2],ui8>,
    %arg2: !torch.vtensor<[],ui8>,
    %arg3: !torch.vtensor<[2],ui8>)
    -> !torch.vtensor<[1,3,2,2],si32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3)
      : (!torch.vtensor<[1,2,3,3],ui8>, !torch.vtensor<[3,2,2,2],ui8>,
         !torch.vtensor<[],ui8>, !torch.vtensor<[2],ui8>)
      -> !torch.vtensor<[1,3,2,2],si32>
  return %0 : !torch.vtensor<[1,3,2,2],si32>
}

// -----

// per-channel w_zero_point with dynamic output channels.
func.func @test_convinteger_per_channel_weight_zp_dynamic_channels(
    %arg0: !torch.vtensor<[1,2,3,3],ui8>,
    %arg1: !torch.vtensor<[?,2,2,2],ui8>,
    %arg2: !torch.vtensor<[],ui8>,
    %arg3: !torch.vtensor<[?],ui8>)
    -> !torch.vtensor<[1,?,2,2],si32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3)
      : (!torch.vtensor<[1,2,3,3],ui8>, !torch.vtensor<[?,2,2,2],ui8>,
         !torch.vtensor<[],ui8>, !torch.vtensor<[?],ui8>)
      -> !torch.vtensor<[1,?,2,2],si32>
  return %0 : !torch.vtensor<[1,?,2,2],si32>
}
