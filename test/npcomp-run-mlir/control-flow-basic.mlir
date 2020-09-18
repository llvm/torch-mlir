// RUN: npcomp-run-mlir %s \
// RUN:   -invoke pow2 \
// RUN:   -arg-value="dense<8.0> : tensor<f32>" \
// RUN:   -shared-libs=%npcomp_runtime_shlib 2>&1 \
// RUN:   | FileCheck %s

// 2^8 == 256
// CHECK: output #0: dense<2.560000e+02> : tensor<f32>
func @pow2(%arg0: tensor<f32>) -> tensor<f32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  // Slight awkwardness: convert the tensor<f32> to an index.
  // TODO: Allow passing plain integers/floats (not tensors) at
  // calling convention boundaries.

  %num_iters_float = extract_element %arg0[] : tensor<f32>
  %num_iters_i32 = fptosi %num_iters_float : f32 to i32
  %num_iters = index_cast %num_iters_i32 : i32 to index

  // Repeatedly add the value to itself %num_iters times.
  %tensor_c1 = constant dense<1.0> : tensor<f32>
  %ret = scf.for %iv = %c0 to %num_iters step %c1 iter_args(%iter = %tensor_c1) -> tensor<f32> {
    %doubled = tcf.add %iter, %iter : (tensor<f32>, tensor<f32>) -> tensor<f32>
    scf.yield %doubled : tensor<f32>
  }
  return %ret : tensor<f32>
}

