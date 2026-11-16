// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s

// An amount known only at runtime is split into max(amount, 0) to pad and
// max(amount, 0) - amount to crop. The crop is checked against the dim at
// runtime.
// CHECK-LABEL: func.func @torch.aten.constant_pad_nd$runtime_amount(
// CHECK:         %[[T:.*]] = torch_c.to_builtin_tensor %{{.*}} : !torch.vtensor<[3,?],f32> -> tensor<3x?xf32>
// CHECK:         %[[AMOUNT:.*]] = arith.index_cast %{{.*}} : i64 to index
// CHECK:         %[[PAD:.*]] = arith.maxsi %[[AMOUNT]], %{{.*}} : index
// CHECK:         %[[CROP:.*]] = arith.subi %[[PAD]], %[[AMOUNT]] : index
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T]], %{{.*}} : tensor<3x?xf32>
// CHECK:         %[[SIZE:.*]] = arith.subi %[[DIM]], %[[CROP]] : index
// CHECK:         %[[FITS:.*]] = arith.cmpi sge, %[[SIZE]], %{{.*}} : index
// CHECK:         cf.assert %[[FITS]], "expected crops that fit the dim"
// CHECK:         %[[SLICE:.*]] = tensor.extract_slice %[[T]][0, %[[CROP]]] [3, %[[SIZE]]] [1, 1]
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
// CHECK-NOT:     cf.assert
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

// A constant crop on a dynamic dim is subtracted from the runtime size and
// checked against it at runtime.
// CHECK-LABEL: func.func @torch.aten.constant_pad_nd$crop_dynamic_dim(
// CHECK:         %[[T:.*]] = torch_c.to_builtin_tensor %{{.*}} : !torch.vtensor<[3,?],f32> -> tensor<3x?xf32>
// CHECK:         %[[DIM:.*]] = tensor.dim %[[T]], %{{.*}} : tensor<3x?xf32>
// CHECK:         %[[CROP:.*]] = arith.constant 2 : index
// CHECK:         %[[SIZE:.*]] = arith.subi %[[DIM]], %[[CROP]] : index
// CHECK:         %[[FITS:.*]] = arith.cmpi sge, %[[SIZE]], %{{.*}} : index
// CHECK:         cf.assert %[[FITS]], "expected crops that fit the dim"
// CHECK:         %[[SLICE:.*]] = tensor.extract_slice %[[T]][0, 2] [3, %[[SIZE]]] [1, 1]
// CHECK:         tensor.pad %[[SLICE]] low[0, 0] high[0, 1]
func.func @torch.aten.constant_pad_nd$crop_dynamic_dim(%arg0: !torch.vtensor<[3,?],f32>) -> !torch.vtensor<[3,?],f32> {
  %int-2 = torch.constant.int -2
  %int1 = torch.constant.int 1
  %float = torch.constant.float 5.000000e-01
  %pad = torch.prim.ListConstruct %int-2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.constant_pad_nd %arg0, %pad, %float : !torch.vtensor<[3,?],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[3,?],f32>
  return %0 : !torch.vtensor<[3,?],f32>
}

// -----

// With strict symbolic shapes, the crop is not checked at runtime.
// CHECK-LABEL: func.func @torch.aten.constant_pad_nd$crop_dynamic_dim_strict(
// CHECK-NOT:     cf.assert
// CHECK:         tensor.extract_slice
func.func @torch.aten.constant_pad_nd$crop_dynamic_dim_strict(%arg0: !torch.vtensor<[3,?],f32>) -> !torch.vtensor<[3,?],f32>
  attributes {torch.assume_strict_symbolic_shapes}
{
  %int-2 = torch.constant.int -2
  %int1 = torch.constant.int 1
  %float = torch.constant.float 5.000000e-01
  %pad = torch.prim.ListConstruct %int-2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.constant_pad_nd %arg0, %pad, %float : !torch.vtensor<[3,?],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[3,?],f32>
  return %0 : !torch.vtensor<[3,?],f32>
}

// -----

// Cropping a whole dim gives a new empty tensor, which is then padded.
// CHECK-LABEL: func.func @torch.aten.constant_pad_nd$crop_whole_dim(
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<3x0xf32>
// CHECK:         tensor.pad %[[EMPTY]] low[0, 0] high[0, 2]
func.func @torch.aten.constant_pad_nd$crop_whole_dim(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,2],f32> {
  %int-4 = torch.constant.int -4
  %int2 = torch.constant.int 2
  %float = torch.constant.float 5.000000e-01
  %pad = torch.prim.ListConstruct %int-4, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.constant_pad_nd %arg0, %pad, %float : !torch.vtensor<[3,4],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// -----

// Cropping a tensor with a static zero-size dim gives a new empty tensor.
// CHECK-LABEL: func.func @torch.aten.constant_pad_nd$crop_zero_size_dim(
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<0x3xf32>
// CHECK:         tensor.pad %[[EMPTY]] low[0, 0] high[0, 2]
func.func @torch.aten.constant_pad_nd$crop_zero_size_dim(%arg0: !torch.vtensor<[0,4],f32>) -> !torch.vtensor<[0,5],f32> {
  %int-1 = torch.constant.int -1
  %int2 = torch.constant.int 2
  %float = torch.constant.float 5.000000e-01
  %pad = torch.prim.ListConstruct %int-1, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.constant_pad_nd %arg0, %pad, %float : !torch.vtensor<[0,4],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[0,5],f32>
  return %0 : !torch.vtensor<[0,5],f32>
}

// -----

// A runtime amount on an empty dim can only pad, so the crop leaves an empty
// tensor instead of slicing it.
// CHECK-LABEL: func.func @torch.aten.constant_pad_nd$runtime_amount_empty_dim(
// CHECK:         cf.assert
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<3x0xf32>
// CHECK:         tensor.pad %[[EMPTY]] low[0, 0] high[0, %{{.*}}]
func.func @torch.aten.constant_pad_nd$runtime_amount_empty_dim(%arg0: !torch.vtensor<[3,0],f32>, %arg1: !torch.int) -> !torch.vtensor<[3,?],f32> {
  %int0 = torch.constant.int 0
  %float = torch.constant.float 0.000000e+00
  %pad = torch.prim.ListConstruct %int0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.constant_pad_nd %arg0, %pad, %float : !torch.vtensor<[3,0],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[3,?],f32>
  return %0 : !torch.vtensor<[3,?],f32>
}

// -----

// A crop larger than the dim is rejected.
func.func @torch.aten.constant_pad_nd$no_convert$crop_exceeds_dim(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,?],f32> {
  %int-5 = torch.constant.int -5
  %int0 = torch.constant.int 0
  %float = torch.constant.float 0.000000e+00
  %pad = torch.prim.ListConstruct %int-5, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  // expected-error @+1 {{failed to legalize operation 'torch.aten.constant_pad_nd'}}
  %0 = torch.aten.constant_pad_nd %arg0, %pad, %float : !torch.vtensor<[3,4],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[3,?],f32>
  return %0 : !torch.vtensor<[3,?],f32>
}
