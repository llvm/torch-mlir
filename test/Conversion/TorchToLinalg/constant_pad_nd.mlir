// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// An amount known only at runtime is split into max(amount, 0) to pad and
// max(amount, 0) - amount to crop.
// CHECK-LABEL: func.func @torch.aten.constant_pad_nd$runtime_amount(
// CHECK:         %[[AMOUNT:.*]] = arith.index_cast %{{.*}} : i64 to index
// CHECK:         %[[PAD:.*]] = arith.maxsi %[[AMOUNT]], %{{.*}} : index
// CHECK:         %[[CROP:.*]] = arith.subi %[[PAD]], %[[AMOUNT]] : index
// CHECK:         %[[SLICE:.*]] = tensor.extract_slice %{{.*}}[0, %[[CROP]]] [3, %{{.*}}] [1, 1]
// CHECK:         tensor.pad %[[SLICE]] low[0, %[[PAD]]] high[0, 0]
func.func @torch.aten.constant_pad_nd$runtime_amount(%arg0: !torch.vtensor<[3,?],f32>, %arg1: !torch.int) -> !torch.vtensor<[3,?],f32> {
  %int0 = torch.constant.int 0
  %float = torch.constant.float 0.000000e+00
  %pad = torch.prim.ListConstruct %arg1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.constant_pad_nd %arg0, %pad, %float : !torch.vtensor<[3,?],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[3,?],f32>
  return %0 : !torch.vtensor<[3,?],f32>
}

// -----

// Non-negative amounts only pad.
// CHECK-LABEL: func.func @torch.aten.constant_pad_nd$pad_only(
// CHECK-NOT:     tensor.extract_slice
// CHECK:         tensor.pad %{{.*}} low[0, 1] high[1, 2]
func.func @torch.aten.constant_pad_nd$pad_only(%arg0: !torch.vtensor<[4,6],f32>) -> !torch.vtensor<[5,9],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %float = torch.constant.float 5.000000e-01
  %pad = torch.prim.ListConstruct %int1, %int2, %int0, %int1 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.constant_pad_nd %arg0, %pad, %float : !torch.vtensor<[4,6],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[5,9],f32>
  return %0 : !torch.vtensor<[5,9],f32>
}

// -----

// Negative amounts crop with tensor.extract_slice before tensor.pad pads the rest.
// CHECK-LABEL: func.func @torch.aten.constant_pad_nd$crop_and_pad(
// CHECK:         %[[SLICE:.*]] = tensor.extract_slice %{{.*}}[0, 2] [3, 4] [1, 1] : tensor<4x6xf32> to tensor<3x4xf32>
// CHECK:         tensor.pad %[[SLICE]] low[1, 0] high[0, 3]
func.func @torch.aten.constant_pad_nd$crop_and_pad(%arg0: !torch.vtensor<[4,6],f32>) -> !torch.vtensor<[4,7],f32> {
  %int-2 = torch.constant.int -2
  %int3 = torch.constant.int 3
  %int1 = torch.constant.int 1
  %int-1 = torch.constant.int -1
  %float = torch.constant.float 5.000000e-01
  %pad = torch.prim.ListConstruct %int-2, %int3, %int1, %int-1 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.constant_pad_nd %arg0, %pad, %float : !torch.vtensor<[4,6],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[4,7],f32>
  return %0 : !torch.vtensor<[4,7],f32>
}

// -----

// Cropping a whole dim gives an empty slice at offset 0, which is then padded.
// CHECK-LABEL: func.func @torch.aten.constant_pad_nd$crop_whole_dim(
// CHECK:         %[[SLICE:.*]] = tensor.extract_slice %{{.*}}[0, 0] [3, 0] [1, 1] : tensor<3x4xf32> to tensor<3x0xf32>
// CHECK:         tensor.pad %[[SLICE]] low[0, 0] high[0, 2]
func.func @torch.aten.constant_pad_nd$crop_whole_dim(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,2],f32> {
  %int-4 = torch.constant.int -4
  %int2 = torch.constant.int 2
  %float = torch.constant.float 5.000000e-01
  %pad = torch.prim.ListConstruct %int-4, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.constant_pad_nd %arg0, %pad, %float : !torch.vtensor<[3,4],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// -----

// A crop larger than the dim is rejected.
func.func @negative_torch.aten.constant_pad_nd$crop_exceeds_dim(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,?],f32> {
  %int-5 = torch.constant.int -5
  %int0 = torch.constant.int 0
  %float = torch.constant.float 0.000000e+00
  %pad = torch.prim.ListConstruct %int-5, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @+1 {{failed to legalize operation 'torch.aten.constant_pad_nd'}}
  %0 = torch.aten.constant_pad_nd %arg0, %pad, %float : !torch.vtensor<[3,4],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[3,?],f32>
  return %0 : !torch.vtensor<[3,?],f32>
}
