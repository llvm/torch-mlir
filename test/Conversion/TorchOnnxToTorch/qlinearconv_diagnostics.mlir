// RUN: torch-mlir-opt %s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch

// w_scale/w_zero_point are neither per-tensor nor per-output-channel (3 != 4).
func.func @test_qlinearconv_weight_scale_size_mismatch(
    %arg0: !torch.vtensor<[1,2,5,5],ui8>,
    %arg1: !torch.vtensor<[],f32>,
    %arg2: !torch.vtensor<[],ui8>,
    %arg3: !torch.vtensor<[4,2,3,3],si8>,
    %arg4: !torch.vtensor<[3],f32>,
    %arg5: !torch.vtensor<[3],si8>,
    %arg6: !torch.vtensor<[],f32>,
    %arg7: !torch.vtensor<[],ui8>)
    -> !torch.vtensor<[1,4,3,3],ui8>
    attributes {torch.onnx_meta.opset_version = 10 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.QLinearConv"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
      : (!torch.vtensor<[1,2,5,5],ui8>, !torch.vtensor<[],f32>,
         !torch.vtensor<[],ui8>, !torch.vtensor<[4,2,3,3],si8>,
         !torch.vtensor<[3],f32>, !torch.vtensor<[3],si8>,
         !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>)
      -> !torch.vtensor<[1,4,3,3],ui8>
  return %0 : !torch.vtensor<[1,4,3,3],ui8>
}

// -----

// w_scale is per-channel but w_zero_point is per-tensor (mixed granularity).
func.func @test_qlinearconv_weight_scale_zp_shape_mismatch(
    %arg0: !torch.vtensor<[1,2,5,5],ui8>,
    %arg1: !torch.vtensor<[],f32>,
    %arg2: !torch.vtensor<[],ui8>,
    %arg3: !torch.vtensor<[4,2,3,3],si8>,
    %arg4: !torch.vtensor<[4],f32>,
    %arg5: !torch.vtensor<[1],si8>,
    %arg6: !torch.vtensor<[],f32>,
    %arg7: !torch.vtensor<[],ui8>)
    -> !torch.vtensor<[1,4,3,3],ui8>
    attributes {torch.onnx_meta.opset_version = 10 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.QLinearConv"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
      : (!torch.vtensor<[1,2,5,5],ui8>, !torch.vtensor<[],f32>,
         !torch.vtensor<[],ui8>, !torch.vtensor<[4,2,3,3],si8>,
         !torch.vtensor<[4],f32>, !torch.vtensor<[1],si8>,
         !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>)
      -> !torch.vtensor<[1,4,3,3],ui8>
  return %0 : !torch.vtensor<[1,4,3,3],ui8>
}

// -----

// w_scale / w_zero_point have a dynamic size, so per-channel cannot be proven.
func.func @test_qlinearconv_weight_scale_dynamic_size(
    %arg0: !torch.vtensor<[1,2,5,5],ui8>,
    %arg1: !torch.vtensor<[],f32>,
    %arg2: !torch.vtensor<[],ui8>,
    %arg3: !torch.vtensor<[4,2,3,3],si8>,
    %arg4: !torch.vtensor<[?],f32>,
    %arg5: !torch.vtensor<[?],si8>,
    %arg6: !torch.vtensor<[],f32>,
    %arg7: !torch.vtensor<[],ui8>)
    -> !torch.vtensor<[1,4,3,3],ui8>
    attributes {torch.onnx_meta.opset_version = 10 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.QLinearConv"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
      : (!torch.vtensor<[1,2,5,5],ui8>, !torch.vtensor<[],f32>,
         !torch.vtensor<[],ui8>, !torch.vtensor<[4,2,3,3],si8>,
         !torch.vtensor<[?],f32>, !torch.vtensor<[?],si8>,
         !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>)
      -> !torch.vtensor<[1,4,3,3],ui8>
  return %0 : !torch.vtensor<[1,4,3,3],ui8>
}

// -----

// x_scale / x_zero_point are not per-tensor.
func.func @test_qlinearconv_input_not_per_tensor(
    %arg0: !torch.vtensor<[1,2,5,5],ui8>,
    %arg1: !torch.vtensor<[2],f32>,
    %arg2: !torch.vtensor<[2],ui8>,
    %arg3: !torch.vtensor<[4,2,3,3],si8>,
    %arg4: !torch.vtensor<[4],f32>,
    %arg5: !torch.vtensor<[4],si8>,
    %arg6: !torch.vtensor<[],f32>,
    %arg7: !torch.vtensor<[],ui8>)
    -> !torch.vtensor<[1,4,3,3],ui8>
    attributes {torch.onnx_meta.opset_version = 10 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.QLinearConv"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
      : (!torch.vtensor<[1,2,5,5],ui8>, !torch.vtensor<[2],f32>,
         !torch.vtensor<[2],ui8>, !torch.vtensor<[4,2,3,3],si8>,
         !torch.vtensor<[4],f32>, !torch.vtensor<[4],si8>,
         !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>)
      -> !torch.vtensor<[1,4,3,3],ui8>
  return %0 : !torch.vtensor<[1,4,3,3],ui8>
}

// -----

// y_scale / y_zero_point are not per-tensor.
func.func @test_qlinearconv_output_not_per_tensor(
    %arg0: !torch.vtensor<[1,2,5,5],ui8>,
    %arg1: !torch.vtensor<[],f32>,
    %arg2: !torch.vtensor<[],ui8>,
    %arg3: !torch.vtensor<[4,2,3,3],si8>,
    %arg4: !torch.vtensor<[4],f32>,
    %arg5: !torch.vtensor<[4],si8>,
    %arg6: !torch.vtensor<[4],f32>,
    %arg7: !torch.vtensor<[4],ui8>)
    -> !torch.vtensor<[1,4,3,3],ui8>
    attributes {torch.onnx_meta.opset_version = 10 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.QLinearConv"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
      : (!torch.vtensor<[1,2,5,5],ui8>, !torch.vtensor<[],f32>,
         !torch.vtensor<[],ui8>, !torch.vtensor<[4,2,3,3],si8>,
         !torch.vtensor<[4],f32>, !torch.vtensor<[4],si8>,
         !torch.vtensor<[4],f32>, !torch.vtensor<[4],ui8>)
      -> !torch.vtensor<[1,4,3,3],ui8>
  return %0 : !torch.vtensor<[1,4,3,3],ui8>
}
