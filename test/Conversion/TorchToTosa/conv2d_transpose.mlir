// RUN: torch-mlir-opt <%s -convert-torch-to-tosa -split-input-file | FileCheck %s

// The lowering now legalizes transpose convolutions into the TOSA dialect.
// Verify that we emit tosa.transpose_conv2d with the expected reshapes/
// permutations.

// CHECK-LABEL: func.func @convTransposeNegativeEffectivePadding
// CHECK-SAME:  %[[INPUT:.*]]: !torch.vtensor<[1,64,1,100],f32>) -> !torch.vtensor<[1,64,2,200],f32> {
// CHECK:  %[[IN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[INPUT]] : !torch.vtensor<[1,64,1,100],f32> -> tensor<1x64x1x100xf32>
// CHECK:  %[[WEIGHT:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<64x64x3x3xf32>}> : () -> tensor<64x64x3x3xf32>
// CHECK:  %[[BIAS:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
// CHECK:  %[[ZP0:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:  %[[ZP1:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:  %[[TRANS_IN:.*]] = tosa.transpose %[[IN_TENSOR]] {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x64x1x100xf32>) -> tensor<1x1x100x64xf32>
// CHECK:  %[[W_OHWI:.*]] = tosa.transpose %[[WEIGHT]] {perms = array<i32: 1, 2, 3, 0>} : (tensor<64x64x3x3xf32>) -> tensor<64x3x3x64xf32>
// CHECK:  %[[TCONV:.*]] = tosa.transpose_conv2d %[[TRANS_IN]], %[[W_OHWI]], %[[BIAS]], %[[ZP0]], %[[ZP1]] {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x1x100x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x3x201x64xf32>
// CHECK-DAG: %[[START_SHAPE:.*]] = tosa.const_shape  {values = dense<[0, 1, 1, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[SLICE_SHAPE:.*]] = tosa.const_shape  {values = dense<[1, 2, 200, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[SLICE_0:.*]] = tosa.slice %[[TCONV]], %[[START_SHAPE]], %[[SLICE_SHAPE]] : (tensor<1x3x201x64xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x2x200x64xf32>
// CHECK:  %[[TRANS_OUT:.*]] = tosa.transpose %[[SLICE_0]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x2x200x64xf32>) -> tensor<1x64x2x200xf32>
// CHECK:  %[[RESULT:.*]] = torch_c.from_builtin_tensor %[[TRANS_OUT]] : tensor<1x64x2x200xf32> -> !torch.vtensor<[1,64,2,200],f32>
// CHECK:  return %[[RESULT]] : !torch.vtensor<[1,64,2,200],f32>
// CHECK: }
func.func @convTransposeNegativeEffectivePadding(%input: !torch.vtensor<[1,64,1,100],f32>) -> !torch.vtensor<[1,64,2,200],f32> {
  %true = torch.constant.bool true
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %weight = torch.vtensor.literal(dense<0.0> : tensor<64x64x3x3xf32>) : !torch.vtensor<[64,64,3,3],f32>
  %bias = torch.vtensor.literal(dense<0.0> : tensor<64xf32>) : !torch.vtensor<[64],f32>
  %stride = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1x1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %output = torch.aten.convolution %input, %weight, %bias, %stride, %int1x1, %int1x1, %true, %int1x1, %int1 : !torch.vtensor<[1,64,1,100],f32>, !torch.vtensor<[64,64,3,3],f32>, !torch.vtensor<[64],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,64,2,200],f32>
  return %output : !torch.vtensor<[1,64,2,200],f32>
}


// -----
// COM: Test case for transpose convolution with positive effective padding to verify no slice operation is emitted
// CHECK-LABEL: func.func @convTransposePositiveEffectivePadding
// CHECK: tosa.transpose_conv2d
// CHECK-NOT: tosa.slice
func.func @convTransposePositiveEffectivePadding(%arg0: !torch.vtensor<[1,2,4,4],f32>, %arg1: !torch.vtensor<[2,2,3,3],f32>) -> !torch.vtensor<[1,2,10,10],f32> {
    %true = torch.constant.bool true
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %none = torch.constant.none
    %int2 = torch.constant.int 2
    %0 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.aten.convolution %arg0, %arg1, %none, %0, %1, %2, %true, %3, %int1 : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[2,2,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,2,10,10],f32>
    return %4 : !torch.vtensor<[1,2,10,10],f32>
  }
