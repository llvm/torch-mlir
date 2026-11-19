// RUN: torch-mlir-opt %s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch

func.func @simplified_layer_normalization_axis_out_of_range(%x: !torch.vtensor<[2,4,8],f32>, %scale: !torch.vtensor<[8],f32>) -> !torch.vtensor<[2,4,8],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.SimplifiedLayerNormalization"(%x, %scale) {torch.onnx.axis = -4 : si64} : (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[8],f32>) -> !torch.vtensor<[2,4,8],f32>
  return %0 : !torch.vtensor<[2,4,8],f32>
}

// -----

func.func @simplified_layer_normalization_extra_result(%x: !torch.vtensor<[2,4,8],f32>, %scale: !torch.vtensor<[8],f32>) -> !torch.vtensor<[2,4,8],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:3 = torch.operator "onnx.SimplifiedLayerNormalization"(%x, %scale) : (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[8],f32>) -> (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],f32>, !torch.vtensor<[2,4,1],f32>)
  return %0#0 : !torch.vtensor<[2,4,8],f32>
}

// -----

func.func @simplified_layer_normalization_non_tensor_inv_std_var(%x: !torch.vtensor<[2,4,8],f32>, %scale: !torch.vtensor<[8],f32>) -> !torch.vtensor<[2,4,8],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:2 = torch.operator "onnx.SimplifiedLayerNormalization"(%x, %scale) : (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[8],f32>) -> (!torch.vtensor<[2,4,8],f32>, !torch.float)
  return %0#0 : !torch.vtensor<[2,4,8],f32>
}

// -----

func.func @simplified_layer_normalization_output_without_dtype(%x: !torch.vtensor<[2,4,8],f32>, %scale: !torch.vtensor<[8],f32>) -> !torch.vtensor<[2,4,8],unk>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.SimplifiedLayerNormalization"(%x, %scale) : (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[8],f32>) -> !torch.vtensor<[2,4,8],unk>
  return %0 : !torch.vtensor<[2,4,8],unk>
}

// -----

func.func @simplified_layer_normalization_inv_std_var_without_dtype(%x: !torch.vtensor<[2,4,8],f32>, %scale: !torch.vtensor<[8],f32>) -> !torch.vtensor<[2,4,1],unk>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:2 = torch.operator "onnx.SimplifiedLayerNormalization"(%x, %scale) : (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[8],f32>) -> (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],unk>)
  return %0#1 : !torch.vtensor<[2,4,1],unk>
}

// -----

func.func @simplified_layer_normalization_unsupported_stash_type(%x: !torch.vtensor<[2,4,8],f32>, %scale: !torch.vtensor<[8],f32>) -> !torch.vtensor<[2,4,8],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.SimplifiedLayerNormalization"(%x, %scale) {torch.onnx.stash_type = 0 : si64} : (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[8],f32>) -> !torch.vtensor<[2,4,8],f32>
  return %0 : !torch.vtensor<[2,4,8],f32>
}

// -----

func.func @simplified_layer_normalization_input_without_dtype(%x: !torch.vtensor<[2,4,8],unk>, %scale: !torch.vtensor<[8],f32>) -> !torch.vtensor<[2,4,8],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.SimplifiedLayerNormalization"(%x, %scale) : (!torch.vtensor<[2,4,8],unk>, !torch.vtensor<[8],f32>) -> !torch.vtensor<[2,4,8],f32>
  return %0 : !torch.vtensor<[2,4,8],f32>
}

// -----

func.func @simplified_layer_normalization_scale_broadcasts_beyond_input(%x: !torch.vtensor<[2,4,8],f32>, %scale: !torch.vtensor<[3,1,1,8],f32>) -> !torch.vtensor<[2,4,8],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.SimplifiedLayerNormalization"(%x, %scale) : (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[3,1,1,8],f32>) -> !torch.vtensor<[2,4,8],f32>
  return %0 : !torch.vtensor<[2,4,8],f32>
}

// -----

func.func @simplified_layer_normalization_incompatible_inv_std_var_shape(%x: !torch.vtensor<[2,4,8],f32>, %scale: !torch.vtensor<[4,8],f32>) -> !torch.vtensor<[2,4,8],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:2 = torch.operator "onnx.SimplifiedLayerNormalization"(%x, %scale) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[4,8],f32>) -> (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,1,8],f32>)
  return %0#0 : !torch.vtensor<[2,4,8],f32>
}

// -----

func.func @simplified_layer_normalization_incompatible_output_shape(%x: !torch.vtensor<[2,4,8],f32>, %scale: !torch.vtensor<[8],f32>) -> !torch.vtensor<[2,4,9],f32>
    attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.SimplifiedLayerNormalization"(%x, %scale) : (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[8],f32>) -> !torch.vtensor<[2,4,9],f32>
  return %0 : !torch.vtensor<[2,4,9],f32>
}
