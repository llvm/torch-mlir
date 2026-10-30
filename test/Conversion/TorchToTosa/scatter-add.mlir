// RUN: torch-mlir-opt %s -convert-torch-to-tosa -split-input-file -verify-diagnostics | FileCheck %s

// Repeated indices must contribute to the same destination.
// CHECK-LABEL: func.func @scatter_add_1d(
// CHECK: tosa.equal
// CHECK: tosa.select
// CHECK: tosa.reduce_sum {{.*}} axis(0)
// CHECK: tosa.add
// CHECK: torch_c.from_builtin_tensor
func.func @scatter_add_1d(%self: !torch.vtensor<[4],si32>, %index: !torch.vtensor<[4],si64>, %src: !torch.vtensor<[4],si32>) -> !torch.vtensor<[4],si32> {
  %dim = torch.constant.int 0
  %result = torch.aten.scatter_add %self, %dim, %index, %src : !torch.vtensor<[4],si32>, !torch.int, !torch.vtensor<[4],si64>, !torch.vtensor<[4],si32> -> !torch.vtensor<[4],si32>
  return %result : !torch.vtensor<[4],si32>
}

// -----

// A negative dim is normalized, and src is sliced to the index shape.
// CHECK-LABEL: func.func @scatter_add_2d(
// CHECK: tosa.slice {{.*}} : (tensor<2x4xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x3xf32>
// CHECK: tosa.equal {{.*}} : (tensor<6x1xi32>, tensor<1x8xi32>) -> tensor<6x8xi1>
// CHECK: tosa.select
// CHECK: tosa.reduce_sum {{.*}} axis(0)
// CHECK: tosa.add
func.func @scatter_add_2d(%self: !torch.vtensor<[2,4],f32>, %index: !torch.vtensor<[2,3],si64>, %src: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %dim = torch.constant.int -1
  %result = torch.aten.scatter_add %self, %dim, %index, %src : !torch.vtensor<[2,4],f32>, !torch.int, !torch.vtensor<[2,3],si64>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// -----

// CHECK-LABEL: func.func @scatter_add_3d_i64(
// CHECK: tosa.equal {{.*}} -> tensor<4x12xi1>
// CHECK: tosa.reduce_sum {{.*}} -> tensor<1x12xi64>
// CHECK: tosa.add {{.*}} -> tensor<1x12xi64>
func.func @scatter_add_3d_i64(%self: !torch.vtensor<[2,3,2],si64>, %index: !torch.vtensor<[2,2,1],si64>, %src: !torch.vtensor<[2,3,2],si64>) -> !torch.vtensor<[2,3,2],si64> {
  %dim = torch.constant.int -2
  %result = torch.aten.scatter_add %self, %dim, %index, %src : !torch.vtensor<[2,3,2],si64>, !torch.int, !torch.vtensor<[2,2,1],si64>, !torch.vtensor<[2,3,2],si64> -> !torch.vtensor<[2,3,2],si64>
  return %result : !torch.vtensor<[2,3,2],si64>
}

// -----

// An empty index leaves self unchanged and does not need a reduction.
// CHECK-LABEL: func.func @scatter_add_empty(
// CHECK-NOT: tosa.reduce_sum
// CHECK: return
func.func @scatter_add_empty(%self: !torch.vtensor<[2,4],f32>, %index: !torch.vtensor<[2,0],si64>, %src: !torch.vtensor<[2,0],f32>) -> !torch.vtensor<[2,4],f32> {
  %dim = torch.constant.int 1
  %result = torch.aten.scatter_add %self, %dim, %index, %src : !torch.vtensor<[2,4],f32>, !torch.int, !torch.vtensor<[2,0],si64>, !torch.vtensor<[2,0],f32> -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// -----

// CHECK-LABEL: func.func @scatter_add_1d_i32_indices(
// CHECK: tosa.equal {{.*}} : (tensor<4x1xi32>, tensor<1x4xi32>) -> tensor<4x4xi1>
// CHECK: tosa.select
// CHECK: tosa.reduce_sum {{.*}} axis(0)
// CHECK: tosa.add
func.func @scatter_add_1d_i32_indices(%self: !torch.vtensor<[4],f32>, %index: !torch.vtensor<[4],si32>, %src: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
  %dim = torch.constant.int 0
  %result = torch.aten.scatter_add %self, %dim, %index, %src : !torch.vtensor<[4],f32>, !torch.int, !torch.vtensor<[4],si32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[4],f32>
  return %result : !torch.vtensor<[4],f32>
}

// -----

// Keep the sum and self addition in f32. For self=[-40000], index=[0,0],
// and src=[40000,40000,0], reducing in f16 would overflow before adding self.
// CHECK-LABEL: func.func @scatter_add_f16(
// CHECK: %[[SLICE:.*]] = tosa.slice {{.*}} -> tensor<2xf16>
// CHECK: %[[SRC:.*]] = tosa.cast %[[SLICE]]{{.*}} : (tensor<2xf16>) -> tensor<2xf32>
// CHECK: %[[UPDATES:.*]] = tosa.reshape %[[SRC]], {{.*}} -> tensor<2x1xf32>
// CHECK: %[[SELECTED:.*]] = tosa.select {{.*}}, %[[UPDATES]], {{.*}} -> tensor<2x1xf32>
// CHECK: %[[SUM:.*]] = tosa.reduce_sum %[[SELECTED]] axis(0) : (tensor<2x1xf32>) -> tensor<1x1xf32>
// CHECK: %[[SELF:.*]] = tosa.cast %{{.*}} : (tensor<1xf16>) -> tensor<1xf32>
// CHECK: %[[FLAT_SELF:.*]] = tosa.reshape %[[SELF]], {{.*}} -> tensor<1x1xf32>
// CHECK: %[[ADDED:.*]] = tosa.add %[[FLAT_SELF]], %[[SUM]] : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: %[[RESULT:.*]] = tosa.reshape %[[ADDED]], {{.*}} -> tensor<1xf32>
// CHECK: tosa.cast %[[RESULT]]{{.*}} : (tensor<1xf32>) -> tensor<1xf16>
func.func @scatter_add_f16(%self: !torch.vtensor<[1],f16>, %index: !torch.vtensor<[2],si64>, %src: !torch.vtensor<[3],f16>) -> !torch.vtensor<[1],f16> {
  %dim = torch.constant.int 0
  %result = torch.aten.scatter_add %self, %dim, %index, %src : !torch.vtensor<[1],f16>, !torch.int, !torch.vtensor<[2],si64>, !torch.vtensor<[3],f16> -> !torch.vtensor<[1],f16>
  return %result : !torch.vtensor<[1],f16>
}

// -----

// BF16 inputs also use f32 accumulation and narrow only after adding self.
// CHECK-LABEL: func.func @scatter_add_bf16(
// CHECK: %[[SRC:.*]] = tosa.cast %{{.*}} : (tensor<3xbf16>) -> tensor<3xf32>
// CHECK: %[[UPDATES:.*]] = tosa.reshape %[[SRC]], {{.*}} -> tensor<3x1xf32>
// CHECK: %[[SELECTED:.*]] = tosa.select {{.*}}, %[[UPDATES]], {{.*}} -> tensor<3x2xf32>
// CHECK: %[[SUM:.*]] = tosa.reduce_sum %[[SELECTED]] axis(0) : (tensor<3x2xf32>) -> tensor<1x2xf32>
// CHECK: %[[SELF:.*]] = tosa.cast %{{.*}} : (tensor<2xbf16>) -> tensor<2xf32>
// CHECK: %[[FLAT_SELF:.*]] = tosa.reshape %[[SELF]], {{.*}} -> tensor<1x2xf32>
// CHECK: %[[ADDED:.*]] = tosa.add %[[FLAT_SELF]], %[[SUM]] : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
// CHECK: %[[RESULT:.*]] = tosa.reshape %[[ADDED]], {{.*}} -> tensor<2xf32>
// CHECK: tosa.cast %[[RESULT]]{{.*}} : (tensor<2xf32>) -> tensor<2xbf16>
func.func @scatter_add_bf16(%self: !torch.vtensor<[2],bf16>, %index: !torch.vtensor<[3],si64>, %src: !torch.vtensor<[3],bf16>) -> !torch.vtensor<[2],bf16> {
  %dim = torch.constant.int 0
  %result = torch.aten.scatter_add %self, %dim, %index, %src : !torch.vtensor<[2],bf16>, !torch.int, !torch.vtensor<[3],si64>, !torch.vtensor<[3],bf16> -> !torch.vtensor<[2],bf16>
  return %result : !torch.vtensor<[2],bf16>
}

// -----

// The product of updates and destinations may equal the 1M-element limit.
// CHECK-LABEL: func.func @scatter_add_at_size_limit(
// CHECK: tosa.select {{.*}} -> tensor<1024x1024xf32>
// CHECK: tosa.reduce_sum {{.*}} -> tensor<1x1024xf32>
func.func @scatter_add_at_size_limit(%self: !torch.vtensor<[1024],f32>, %index: !torch.vtensor<[1024],si64>, %src: !torch.vtensor<[1024],f32>) -> !torch.vtensor<[1024],f32> {
  %dim = torch.constant.int 0
  %result = torch.aten.scatter_add %self, %dim, %index, %src : !torch.vtensor<[1024],f32>, !torch.int, !torch.vtensor<[1024],si64>, !torch.vtensor<[1024],f32> -> !torch.vtensor<[1024],f32>
  return %result : !torch.vtensor<[1024],f32>
}

// -----

// Switch to sort and segmented sum above the dense intermediate limit.
// CHECK-LABEL: func.func @scatter_add_above_size_limit(
// CHECK-NOT: tosa.reduce_sum
// CHECK: tosa.bitwise_and
// CHECK: tosa.scatter
// CHECK: tosa.gather
// CHECK: tosa.scatter {{.*}} -> tensor<1x2049x1xf32>
// CHECK: tosa.slice {{.*}} -> tensor<1x1024x1xf32>
// CHECK: tosa.add {{.*}} -> tensor<1x1024xf32>
// CHECK-NOT: tosa.reduce_sum
// CHECK: return
func.func @scatter_add_above_size_limit(%self: !torch.vtensor<[1024],f32>, %index: !torch.vtensor<[1025],si64>, %src: !torch.vtensor<[1025],f32>) -> !torch.vtensor<[1024],f32> {
  %dim = torch.constant.int 0
  %result = torch.aten.scatter_add %self, %dim, %index, %src : !torch.vtensor<[1024],f32>, !torch.int, !torch.vtensor<[1025],si64>, !torch.vtensor<[1025],f32> -> !torch.vtensor<[1024],f32>
  return %result : !torch.vtensor<[1024],f32>
}

// -----

// This product would overflow i64, and source positions do not fit in i32.
// Reject it before allocating per-update or per-destination constants.
func.func @scatter_add_size_product_overflow(%self: !torch.vtensor<[1073741824],f32>, %index: !torch.vtensor<[17179869184],si64>, %src: !torch.vtensor<[17179869184],f32>) -> !torch.vtensor<[1073741824],f32> {
  %dim = torch.constant.int 0
  // expected-error @+1 {{failed to legalize operation 'torch.aten.scatter_add'}}
  %result = torch.aten.scatter_add %self, %dim, %index, %src : !torch.vtensor<[1073741824],f32>, !torch.int, !torch.vtensor<[17179869184],si64>, !torch.vtensor<[17179869184],f32> -> !torch.vtensor<[1073741824],f32>
  return %result : !torch.vtensor<[1073741824],f32>
}

// -----

// Each count fits in i32, but their sum does not fit the scratch index range.
func.func @scatter_add_scratch_indices_overflow(%self: !torch.vtensor<[2147483646],f32>, %index: !torch.vtensor<[2],si32>, %src: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2147483646],f32> {
  %dim = torch.constant.int 0
  // expected-error @+1 {{failed to legalize operation 'torch.aten.scatter_add'}}
  %result = torch.aten.scatter_add %self, %dim, %index, %src : !torch.vtensor<[2147483646],f32>, !torch.int, !torch.vtensor<[2],si32>, !torch.vtensor<[2],f32> -> !torch.vtensor<[2147483646],f32>
  return %result : !torch.vtensor<[2147483646],f32>
}

// -----

// A large vector uses linear-size scratch storage.
// CHECK-LABEL: func.func @scatter_add_large_vector(
// CHECK: tosa.bitwise_and
// CHECK: tosa.gather {{.*}} -> tensor<1x209985x1xf32>
// CHECK: tosa.scatter {{.*}} -> tensor<1x219985x1xf32>
// CHECK: tosa.slice {{.*}} -> tensor<1x10000x1xf32>
// CHECK: tosa.add {{.*}} -> tensor<1x10000xf32>
func.func @scatter_add_large_vector(%self: !torch.vtensor<[10000],f32>, %index: !torch.vtensor<[209985],si32>, %src: !torch.vtensor<[209985],f32>) -> !torch.vtensor<[10000],f32> {
  %dim = torch.constant.int 0
  %result = torch.aten.scatter_add %self, %dim, %index, %src : !torch.vtensor<[10000],f32>, !torch.int, !torch.vtensor<[209985],si32>, !torch.vtensor<[209985],f32> -> !torch.vtensor<[10000],f32>
  return %result : !torch.vtensor<[10000],f32>
}

// -----

// Share negative-dim handling, source slicing, and f32 accumulation with the
// dense path. The segmented scan and final addition stay in f32.
// CHECK-LABEL: func.func @scatter_add_sorted_f16(
// CHECK: tosa.slice {{.*}} -> tensor<2x513xf16>
// CHECK: tosa.cast {{.*}} : (tensor<2x513xf16>) -> tensor<2x513xf32>
// CHECK: tosa.gather {{.*}} -> tensor<1x1026x1xf32>
// CHECK: tosa.scatter {{.*}} -> tensor<1x2050x1xf32>
// CHECK: tosa.add {{.*}} -> tensor<1x1024xf32>
// CHECK: tosa.cast {{.*}} : (tensor<2x512xf32>) -> tensor<2x512xf16>
func.func @scatter_add_sorted_f16(%self: !torch.vtensor<[2,512],f16>, %index: !torch.vtensor<[2,513],si64>, %src: !torch.vtensor<[2,514],f16>) -> !torch.vtensor<[2,512],f16> {
  %dim = torch.constant.int -1
  %result = torch.aten.scatter_add %self, %dim, %index, %src : !torch.vtensor<[2,512],f16>, !torch.int, !torch.vtensor<[2,513],si64>, !torch.vtensor<[2,514],f16> -> !torch.vtensor<[2,512],f16>
  return %result : !torch.vtensor<[2,512],f16>
}

// -----

// With one update, sorting and segmented scanning are unnecessary.
// CHECK-LABEL: func.func @scatter_add_sorted_single_update(
// CHECK-NOT: tosa.bitwise_and
// CHECK: tosa.gather {{.*}} -> tensor<1x1x1xi64>
// CHECK: tosa.scatter {{.*}} -> tensor<1x1048578x1xi64>
// CHECK-NOT: tosa.bitwise_and
// CHECK: return
func.func @scatter_add_sorted_single_update(%self: !torch.vtensor<[1048577],si64>, %index: !torch.vtensor<[1],si32>, %src: !torch.vtensor<[1],si64>) -> !torch.vtensor<[1048577],si64> {
  %dim = torch.constant.int 0
  %result = torch.aten.scatter_add %self, %dim, %index, %src : !torch.vtensor<[1048577],si64>, !torch.int, !torch.vtensor<[1],si32>, !torch.vtensor<[1],si64> -> !torch.vtensor<[1048577],si64>
  return %result : !torch.vtensor<[1048577],si64>
}

// -----

// BF16 values also use f32 for the segmented scan and final addition.
// CHECK-LABEL: func.func @scatter_add_sorted_bf16(
// CHECK: tosa.cast {{.*}} : (tensor<17xbf16>) -> tensor<17xf32>
// CHECK: tosa.gather {{.*}} -> tensor<1x17x1xf32>
// CHECK: tosa.scatter {{.*}} -> tensor<1x65554x1xf32>
// CHECK: tosa.add {{.*}} -> tensor<1x65537xf32>
// CHECK: tosa.cast {{.*}} : (tensor<65537xf32>) -> tensor<65537xbf16>
func.func @scatter_add_sorted_bf16(%self: !torch.vtensor<[65537],bf16>, %index: !torch.vtensor<[17],si64>, %src: !torch.vtensor<[17],bf16>) -> !torch.vtensor<[65537],bf16> {
  %dim = torch.constant.int 0
  %result = torch.aten.scatter_add %self, %dim, %index, %src : !torch.vtensor<[65537],bf16>, !torch.int, !torch.vtensor<[17],si64>, !torch.vtensor<[17],bf16> -> !torch.vtensor<[65537],bf16>
  return %result : !torch.vtensor<[65537],bf16>
}
