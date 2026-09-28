// RUN: torch-mlir-opt %s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch

// LRN reshapes the input to [N, 1, C, H, W_collapsed]. aten.view infers at most
// one dimension, so at most one of the batch, spatial (H), and collapsed
// trailing dims may be dynamic. With both the batch and a trailing dim dynamic
// the reshape would need two inferred dimensions, so the conversion bails out.
func.func @test_lrn_multiple_dynamic_dims(%arg0: !torch.vtensor<[?,96,55,?],f32>) -> !torch.vtensor<[?,96,55,?],f32> attributes {torch.onnx_meta.opset_version = 17 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator' that was explicitly marked illegal}}
  %0 = torch.operator "onnx.LRN"(%arg0) {torch.onnx.size = 5 : si64} : (!torch.vtensor<[?,96,55,?],f32>) -> !torch.vtensor<[?,96,55,?],f32>
  return %0 : !torch.vtensor<[?,96,55,?],f32>
}
