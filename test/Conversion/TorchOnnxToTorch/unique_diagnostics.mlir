// RUN: torch-mlir-opt <%s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch

func.func @test_unique_too_many_results(%arg0: !torch.vtensor<[6],f32>) -> !torch.vtensor<[4],f32> attributes {torch.onnx_meta.opset_version = 11 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:5 = torch.operator "onnx.Unique"(%arg0) {torch.onnx.sorted = 1 : si64} : (!torch.vtensor<[6],f32>) -> (!torch.vtensor<[4],f32>, !torch.vtensor<[4],si64>, !torch.vtensor<[6],si64>, !torch.vtensor<[4],si64>, !torch.vtensor<[4],si64>)
  return %0#0 : !torch.vtensor<[4],f32>
}

// -----

func.func @test_unique_rank0_input(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[?],f32> attributes {torch.onnx_meta.opset_version = 11 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.Unique"(%arg0) {torch.onnx.sorted = 1 : si64} : (!torch.vtensor<[],f32>) -> !torch.vtensor<[?],f32>
  return %0 : !torch.vtensor<[?],f32>
}

// -----

func.func @test_unique_output_rank_does_not_cover_axis(%arg0: !torch.vtensor<[3,3],f32>) -> !torch.vtensor<[?],f32> attributes {torch.onnx_meta.opset_version = 11 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.Unique"(%arg0) {torch.onnx.axis = 1 : si64, torch.onnx.sorted = 1 : si64} : (!torch.vtensor<[3,3],f32>) -> !torch.vtensor<[?],f32>
  return %0 : !torch.vtensor<[?],f32>
}

// -----

func.func @test_unique_indices_dynamic_size_along_axis(%arg0: !torch.vtensor<[?,3],f32>) -> (!torch.vtensor<[?,3],f32>, !torch.vtensor<[?],si64>) attributes {torch.onnx_meta.opset_version = 11 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0:2 = torch.operator "onnx.Unique"(%arg0) {torch.onnx.axis = 0 : si64, torch.onnx.sorted = 1 : si64} : (!torch.vtensor<[?,3],f32>) -> (!torch.vtensor<[?,3],f32>, !torch.vtensor<[?],si64>)
  return %0#0, %0#1 : !torch.vtensor<[?,3],f32>, !torch.vtensor<[?],si64>
}
