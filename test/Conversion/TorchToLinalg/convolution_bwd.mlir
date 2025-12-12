// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -canonicalize -split-input-file -mlir-print-local-scope -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @convolution_backward_input_1x1s_0x0p_1x1d_1g(
// CHECK-SAME:                                                %[[VAL_0:.*]]: !torch.vtensor<[2,16,63,63],f32>, %[[VAL_1:.*]]: !torch.vtensor<[2,128,64,64],f32>,
// CHECK-SAME:                                                %[[VAL_2:.*]]: !torch.vtensor<[16,128,2,2],f32>,
// CHECK-SAME:                                                %[[VAL_3:.*]]: !torch.vtensor<[],f32>) -> (!torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16],f32>) {
func.func @convolution_backward_input_1x1s_0x0p_1x1d_1g(%arg0: !torch.vtensor<[2,16,63,63],f32>, %arg1: !torch.vtensor<[2,128,64,64],f32>, %arg2: !torch.vtensor<[16,128,2,2],f32>, %arg3: !torch.vtensor<[],f32>) -> (!torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16],f32>) {
  // CHECK:           %[[CST1:.*]] = arith.constant 1 : index
  // CHECK:           %[[CST0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:           %[[T1:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[16,128,2,2],f32> -> tensor<16x128x2x2xf32>
  // CHECK:           %[[T0:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,16,63,63],f32> -> tensor<2x16x63x63xf32>
  // CHECK:           %[[W_EMPTY:.*]] = tensor.empty() : tensor<16x128x2x2xf32>
  // CHECK:           %[[W_FILLED:.*]] = linalg.fill ins(%[[CST0]] : f32) outs(%[[W_EMPTY]] : tensor<16x128x2x2xf32>) -> tensor<16x128x2x2xf32>
  // CHECK:           %[[W_REV:.*]] = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[T1]] : tensor<16x128x2x2xf32>) outs(%[[W_FILLED]] : tensor<16x128x2x2xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN_W:.*]]: f32, %[[OUT_W:.*]]: f32):
  // CHECK-NEXT:        %[[I0:.*]] = linalg.index 0 : index
  // CHECK-NEXT:        %[[I1:.*]] = linalg.index 1 : index
  // CHECK-NEXT:        %[[I2:.*]] = linalg.index 2 : index
  // CHECK-NEXT:        %[[I3:.*]] = linalg.index 3 : index
  // CHECK-NEXT:        %[[R2:.*]] = arith.subi %[[CST1]], %[[I2]] : index
  // CHECK-NEXT:        %[[R3:.*]] = arith.subi %[[CST1]], %[[I3]] : index
  // CHECK-NEXT:        %[[EX:.*]] = tensor.extract %[[T1]][%[[I0]], %[[I1]], %[[R2]], %[[R3]]] : tensor<16x128x2x2xf32>
  // CHECK-NEXT:        linalg.yield %[[EX]] : f32
  // CHECK-NEXT:      } -> tensor<16x128x2x2xf32>
  // CHECK:           %[[PAD:.*]] = tensor.pad %[[T0]] low[0, 0, 1, 1] high[0, 0, 1, 1]
  // CHECK:           ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
  // CHECK:             tensor.yield %[[CST0]] : f32
  // CHECK:           } : tensor<2x16x63x63xf32> to tensor<2x16x65x65xf32>
  // CHECK:           %[[OUT_EMPTY:.*]] = tensor.empty() : tensor<2x128x64x64xf32>
  // CHECK:           %[[OUT_FILLED:.*]] = linalg.fill ins(%[[CST0]] : f32) outs(%[[OUT_EMPTY]] : tensor<2x128x64x64xf32>) -> tensor<2x128x64x64xf32>
  // CHECK:           %[[CONV:.*]] = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%[[PAD]], %[[W_REV]] : tensor<2x16x65x65xf32>, tensor<16x128x2x2xf32>) outs(%[[OUT_FILLED]] : tensor<2x128x64x64xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN:.*]]: f32, %[[IN1:.*]]: f32, %[[OUT:.*]]: f32):
  // CHECK-NEXT:        %[[MUL:.*]] = arith.mulf %[[IN]], %[[IN1]] : f32
  // CHECK-NEXT:        %[[ACC:.*]] = arith.addf %[[MUL]], %[[OUT]] : f32
  // CHECK-NEXT:        linalg.yield %[[ACC]] : f32
  // CHECK-NEXT:      } -> tensor<2x128x64x64xf32>
  // CHECK:           %[[IGRAD:.*]] = torch_c.from_builtin_tensor %[[CONV]] : tensor<2x128x64x64xf32> -> !torch.vtensor<[2,128,64,64],f32>
  // CHECK:           %[[SUM_EMPTY:.*]] = tensor.empty() : tensor<16xf32>
  // CHECK:           %[[SUM_FILLED:.*]] = linalg.fill ins(%[[CST0]] : f32) outs(%[[SUM_EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
  // CHECK:           %[[SUM_GEN:.*]] = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>], iterator_types = ["reduction", "parallel", "reduction", "reduction"]} ins(%[[T0]] : tensor<2x16x63x63xf32>) outs(%[[SUM_FILLED]] : tensor<16xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN_B:.*]]: f32, %[[ACC_B:.*]]: f32):
  // CHECK-NEXT:        %[[B_RES:.*]] = arith.addf %[[IN_B]], %[[ACC_B]] : f32
  // CHECK-NEXT:        linalg.yield %[[B_RES]] : f32
  // CHECK-NEXT:      } -> tensor<16xf32>
  // CHECK:           %[[BIAS:.*]] = torch_c.from_builtin_tensor %[[SUM_GEN]] : tensor<16xf32> -> !torch.vtensor<[16],f32>
  // CHECK:           return %[[IGRAD]], %[[BIAS]] : !torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16],f32>
  %true = torch.constant.bool true
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %true, %false, %true : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>
  %result0, %result1, %result2 = torch.aten.convolution_backward %arg0, %arg1, %arg2, %0, %1, %2, %1, %false, %2, %int1, %3 : !torch.vtensor<[2,16,63,63],f32>, !torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16,128,2,2],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int, !torch.list<bool> -> !torch.vtensor<[2,128,64,64],f32>, !torch.none, !torch.vtensor<[16],f32>
  return %result0, %result2 : !torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16],f32>
}

// -----

// CHECK-LABEL:   func.func @convolution_backward_input_1x1ker_1x1s_0x0p_1x1d_1g(
// CHECK-SAME:                                                %[[VAL_0:.*]]: !torch.vtensor<[2,16,64,64],f32>, %[[VAL_1:.*]]: !torch.vtensor<[2,128,64,64],f32>,
// CHECK-SAME:                                                %[[VAL_2:.*]]: !torch.vtensor<[16,128,1,1],f32>,
// CHECK-SAME:                                                %[[VAL_3:.*]]: !torch.vtensor<[],f32>) -> (!torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16],f32>) {
func.func @convolution_backward_input_1x1ker_1x1s_0x0p_1x1d_1g(%arg0: !torch.vtensor<[2,16,64,64],f32>, %arg1: !torch.vtensor<[2,128,64,64],f32>, %arg2: !torch.vtensor<[16,128,1,1],f32>, %arg3: !torch.vtensor<[],f32>) -> (!torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16],f32>) {
  // CHECK:           %[[CST0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:           %[[T1:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[16,128,1,1],f32> -> tensor<16x128x1x1xf32>
  // CHECK:           %[[T0:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,16,64,64],f32> -> tensor<2x16x64x64xf32>
  // CHECK:           %[[OUT_EMPTY:.*]] = tensor.empty() : tensor<2x128x64x64xf32>
  // CHECK:           %[[OUT_FILLED:.*]] = linalg.fill ins(%[[CST0]] : f32) outs(%[[OUT_EMPTY]] : tensor<2x128x64x64xf32>) -> tensor<2x128x64x64xf32>
  // CHECK:           %[[CONV:.*]] = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%[[T0]], %[[T1]] : tensor<2x16x64x64xf32>, tensor<16x128x1x1xf32>) outs(%[[OUT_FILLED]] : tensor<2x128x64x64xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN:.*]]: f32, %[[IN1:.*]]: f32, %[[OUT:.*]]: f32):
  // CHECK-NEXT:        %[[MUL:.*]] = arith.mulf %[[IN]], %[[IN1]] : f32
  // CHECK-NEXT:        %[[ACC:.*]] = arith.addf %[[MUL]], %[[OUT]] : f32
  // CHECK-NEXT:        linalg.yield %[[ACC]] : f32
  // CHECK-NEXT:      } -> tensor<2x128x64x64xf32>
  // CHECK:           %[[IGRAD:.*]] = torch_c.from_builtin_tensor %[[CONV]] : tensor<2x128x64x64xf32> -> !torch.vtensor<[2,128,64,64],f32>
  // CHECK:           %[[SUM_EMPTY:.*]] = tensor.empty() : tensor<16xf32>
  // CHECK:           %[[SUM_FILLED:.*]] = linalg.fill ins(%[[CST0]] : f32) outs(%[[SUM_EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
  // CHECK:           %[[SUM_GEN:.*]] = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>], iterator_types = ["reduction", "parallel", "reduction", "reduction"]} ins(%[[T0]] : tensor<2x16x64x64xf32>) outs(%[[SUM_FILLED]] : tensor<16xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN_B:.*]]: f32, %[[ACC_B:.*]]: f32):
  // CHECK-NEXT:        %[[B_RES:.*]] = arith.addf %[[IN_B]], %[[ACC_B]] : f32
  // CHECK-NEXT:        linalg.yield %[[B_RES]] : f32
  // CHECK-NEXT:      } -> tensor<16xf32>
  // CHECK:           %[[BIAS:.*]] = torch_c.from_builtin_tensor %[[SUM_GEN]] : tensor<16xf32> -> !torch.vtensor<[16],f32>
  // CHECK:           return %[[IGRAD]], %[[BIAS]] : !torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16],f32>
  %true = torch.constant.bool true
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %true, %false, %true : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>
  %result0, %result1, %result2 = torch.aten.convolution_backward %arg0, %arg1, %arg2, %0, %1, %2, %1, %false, %2, %int1, %3 : !torch.vtensor<[2,16,64,64],f32>, !torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16,128,1,1],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int, !torch.list<bool> -> !torch.vtensor<[2,128,64,64],f32>, !torch.none, !torch.vtensor<[16],f32>
  return %result0, %result2 : !torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16],f32>
}

// -----

// CHECK-LABEL:   func.func @convolution_backward_input_2x2s_2x2p_2x2d_1g(
// CHECK-SAME:                                                %[[VAL_0:.*]]: !torch.vtensor<[2,16,33,33],f32>, %[[VAL_1:.*]]: !torch.vtensor<[2,128,64,64],f32>,
// CHECK-SAME:                                                %[[VAL_2:.*]]: !torch.vtensor<[16,128,2,2],f32>,
// CHECK-SAME:                                                %[[VAL_3:.*]]: !torch.vtensor<[],f32>) -> (!torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16],f32>) {
func.func @convolution_backward_input_2x2s_2x2p_2x2d_1g(%arg0: !torch.vtensor<[2,16,33,33],f32>, %arg1: !torch.vtensor<[2,128,64,64],f32>, %arg2: !torch.vtensor<[16,128,2,2],f32>, %arg3: !torch.vtensor<[],f32>) -> (!torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16],f32>) {
  // CHECK:           %[[CST1:.*]] = arith.constant 1 : index
  // CHECK:           %[[CST0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:           %[[T1:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[16,128,2,2],f32> -> tensor<16x128x2x2xf32>
  // CHECK:           %[[T0:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,16,33,33],f32> -> tensor<2x16x33x33xf32>
  // CHECK:           %[[W_EMPTY:.*]] = tensor.empty() : tensor<16x128x2x2xf32>
  // CHECK:           %[[W_FILLED:.*]] = linalg.fill ins(%[[CST0]] : f32) outs(%[[W_EMPTY]] : tensor<16x128x2x2xf32>) -> tensor<16x128x2x2xf32>
  // CHECK:           %[[W_REV:.*]] = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[T1]] : tensor<16x128x2x2xf32>) outs(%[[W_FILLED]] : tensor<16x128x2x2xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN_W:.*]]: f32, %[[OUT_W:.*]]: f32):
  // CHECK-NEXT:        %[[I0:.*]] = linalg.index 0 : index
  // CHECK-NEXT:        %[[I1:.*]] = linalg.index 1 : index
  // CHECK-NEXT:        %[[I2:.*]] = linalg.index 2 : index
  // CHECK-NEXT:        %[[I3:.*]] = linalg.index 3 : index
  // CHECK-NEXT:        %[[R2:.*]] = arith.subi %[[CST1]], %[[I2]] : index
  // CHECK-NEXT:        %[[R3:.*]] = arith.subi %[[CST1]], %[[I3]] : index
  // CHECK-NEXT:        %[[EX:.*]] = tensor.extract %[[T1]][%[[I0]], %[[I1]], %[[R2]], %[[R3]]] : tensor<16x128x2x2xf32>
  // CHECK-NEXT:        linalg.yield %[[EX]] : f32
  // CHECK-NEXT:      } -> tensor<16x128x2x2xf32>
  // CHECK:           %[[SLICE_EMPTY:.*]] = tensor.empty() : tensor<2x16x66x66xf32>
  // CHECK:           %[[SLICE_FILLED:.*]] = linalg.fill ins(%cst : f32) outs(%[[SLICE_EMPTY]] : tensor<2x16x66x66xf32>) -> tensor<2x16x66x66xf32>
  // CHECK:           %[[SLICE:.*]]  = tensor.insert_slice %[[T0]] into %[[SLICE_FILLED]][0, 0, 0, 0] [2, 16, 33, 33] [1, 1, 2, 2] : tensor<2x16x33x33xf32> into tensor<2x16x66x66xf32>
  // CHECK:           %[[OUT_EMPTY:.*]] = tensor.empty() : tensor<2x128x64x64xf32>
  // CHECK:           %[[OUT_FILLED:.*]] = linalg.fill ins(%[[CST0]] : f32) outs(%[[OUT_EMPTY]] : tensor<2x128x64x64xf32>) -> tensor<2x128x64x64xf32>
  // CHECK:           %[[CONV:.*]] = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d5 * 2 + d2, d6 * 2 + d3)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%[[SLICE]], %[[W_REV]] : tensor<2x16x66x66xf32>, tensor<16x128x2x2xf32>) outs(%[[OUT_FILLED]] : tensor<2x128x64x64xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN:.*]]: f32, %[[IN1:.*]]: f32, %[[OUT:.*]]: f32):
  // CHECK-NEXT:        %[[MUL:.*]] = arith.mulf %[[IN]], %[[IN1]] : f32
  // CHECK-NEXT:        %[[ACC:.*]] = arith.addf %[[MUL]], %[[OUT]] : f32
  // CHECK-NEXT:        linalg.yield %[[ACC]] : f32
  // CHECK-NEXT:      } -> tensor<2x128x64x64xf32>
  // CHECK:           %[[IGRAD:.*]] = torch_c.from_builtin_tensor %[[CONV]] : tensor<2x128x64x64xf32> -> !torch.vtensor<[2,128,64,64],f32>
  // CHECK:           %[[SUM_EMPTY:.*]] = tensor.empty() : tensor<16xf32>
  // CHECK:           %[[SUM_FILLED:.*]] = linalg.fill ins(%[[CST0]] : f32) outs(%[[SUM_EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
  // CHECK:           %[[SUM_GEN:.*]] = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>], iterator_types = ["reduction", "parallel", "reduction", "reduction"]} ins(%[[T0]] : tensor<2x16x33x33xf32>) outs(%[[SUM_FILLED]] : tensor<16xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN_B:.*]]: f32, %[[ACC_B:.*]]: f32):
  // CHECK-NEXT:        %[[B_RES:.*]] = arith.addf %[[IN_B]], %[[ACC_B]] : f32
  // CHECK-NEXT:        linalg.yield %[[B_RES]] : f32
  // CHECK-NEXT:      } -> tensor<16xf32>
  // CHECK:           %[[BIAS:.*]] = torch_c.from_builtin_tensor %[[SUM_GEN]] : tensor<16xf32> -> !torch.vtensor<[16],f32>
  // CHECK:           return %[[IGRAD]], %[[BIAS]] : !torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16],f32>
  %true = torch.constant.bool true
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %true, %false, %true : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>
  %result0, %result1, %result2 = torch.aten.convolution_backward %arg0, %arg1, %arg2, %0, %1, %1, %1, %false, %2, %int1, %3 : !torch.vtensor<[2,16,33,33],f32>, !torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16,128,2,2],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int, !torch.list<bool> -> !torch.vtensor<[2,128,64,64],f32>, !torch.none, !torch.vtensor<[16],f32>
  return %result0, %result2 : !torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16],f32>
}

// -----

// CHECK-LABEL:   func.func @convolution_backward_input_2x2s_2x2p_2x2d_4g(
// CHECK-SAME:                                                %[[VAL_0:.*]]: !torch.vtensor<[2,16,33,33],f32>, %[[VAL_1:.*]]: !torch.vtensor<[2,128,64,64],f32>,
// CHECK-SAME:                                                %[[VAL_2:.*]]: !torch.vtensor<[16,32,2,2],f32>,
// CHECK-SAME:                                                %[[VAL_3:.*]]: !torch.vtensor<[],f32>) -> (!torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16],f32>) {
func.func @convolution_backward_input_2x2s_2x2p_2x2d_4g(%arg0: !torch.vtensor<[2,16,33,33],f32>, %arg1: !torch.vtensor<[2,128,64,64],f32>, %arg2: !torch.vtensor<[16,32,2,2],f32>, %arg3: !torch.vtensor<[],f32>) -> (!torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16],f32>) {
  // CHECK:           %[[CST1:.*]] = arith.constant 1 : index
  // CHECK:           %[[CST0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:           %[[T1:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[16,32,2,2],f32> -> tensor<16x32x2x2xf32>
  // CHECK:           %[[T0:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,16,33,33],f32> -> tensor<2x16x33x33xf32>
  // CHECK:           %[[T0_EXP:.*]] = tensor.expand_shape %[[T0]] {{\[\[0\], \[1, 2\], \[3\], \[4\]\]}} output_shape [2, 4, 4, 33, 33] : tensor<2x16x33x33xf32> into tensor<2x4x4x33x33xf32>
  // CHECK:           %[[W_EXP:.*]] = tensor.expand_shape %[[T1]] {{\[\[0, 1\], \[2\], \[3\], \[4\]\]}} output_shape [4, 4, 32, 2, 2] : tensor<16x32x2x2xf32> into tensor<4x4x32x2x2xf32>
  // CHECK:           %[[W_EMPTY:.*]] = tensor.empty() : tensor<4x4x32x2x2xf32>
  // CHECK:           %[[W_FILLED:.*]] = linalg.fill ins(%[[CST0]] : f32) outs(%[[W_EMPTY]] : tensor<4x4x32x2x2xf32>) -> tensor<4x4x32x2x2xf32>
  // CHECK:           %[[W_REV:.*]] = linalg.generic {{.*}} ins(%[[W_EXP]] : tensor<4x4x32x2x2xf32>) outs(%[[W_FILLED]] : tensor<4x4x32x2x2xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN_W:.*]]: f32, %[[OUT_W:.*]]: f32):
  // CHECK-NEXT:        %[[I0:.*]] = linalg.index 0 : index
  // CHECK-NEXT:        %[[I1:.*]] = linalg.index 1 : index
  // CHECK-NEXT:        %[[I2:.*]] = linalg.index 2 : index
  // CHECK-NEXT:        %[[I3:.*]] = linalg.index 3 : index
  // CHECK-NEXT:        %[[I4:.*]] = linalg.index 4 : index
  // CHECK-NEXT:        %[[R3:.*]] = arith.subi %[[CST1]], %[[I3]] : index
  // CHECK-NEXT:        %[[R4:.*]] = arith.subi %[[CST1]], %[[I4]] : index
  // CHECK-NEXT:        %[[EX:.*]] = tensor.extract %[[W_EXP]][%[[I0]], %[[I1]], %[[I2]], %[[R3]], %[[R4]]] : tensor<4x4x32x2x2xf32>
  // CHECK-NEXT:        linalg.yield %[[EX]] : f32
  // CHECK-NEXT:      } -> tensor<4x4x32x2x2xf32>
  // CHECK:           %[[SLICE_EMPTY:.*]] = tensor.empty() : tensor<2x4x4x66x66xf32>
  // CHECK:           %[[SLICE_FILLED:.*]] = linalg.fill ins(%cst : f32) outs(%[[SLICE_EMPTY]] : tensor<2x4x4x66x66xf32>) -> tensor<2x4x4x66x66xf32>
  // CHECK:           %[[SLICE:.*]]  = tensor.insert_slice %[[T0_EXP]] into %[[SLICE_FILLED]][0, 0, 0, 0, 0] [2, 4, 4, 33, 33] [1, 1, 1, 2, 2] : tensor<2x4x4x33x33xf32> into tensor<2x4x4x66x66xf32>
  // CHECK:           %[[OUT_EMPTY:.*]] = tensor.empty() : tensor<2x4x32x64x64xf32>
  // CHECK:           %[[OUT_FILLED:.*]] = linalg.fill ins(%[[CST0]] : f32) outs(%[[OUT_EMPTY]] : tensor<2x4x32x64x64xf32>) -> tensor<2x4x32x64x64xf32>
  // CHECK:           %[[CONV:.*]] = linalg.generic {{.*}} ins(%[[SLICE]], %[[W_REV]] : tensor<2x4x4x66x66xf32>, tensor<4x4x32x2x2xf32>) outs(%[[OUT_FILLED]] : tensor<2x4x32x64x64xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN:.*]]: f32, %[[IN1:.*]]: f32, %[[OUT:.*]]: f32):
  // CHECK-NEXT:        %[[MUL:.*]] = arith.mulf %[[IN]], %[[IN1]] : f32
  // CHECK-NEXT:        %[[ACC:.*]] = arith.addf %[[MUL]], %[[OUT]] : f32
  // CHECK-NEXT:        linalg.yield %[[ACC]] : f32
  // CHECK-NEXT:      } -> tensor<2x4x32x64x64xf32>
  // CHECK:           %[[CONV_COLLAPSED:.*]] = tensor.collapse_shape %[[CONV]] {{\[\[0\], \[1, 2\], \[3\], \[4\]\]}} : tensor<2x4x32x64x64xf32> into tensor<2x128x64x64xf32>
  // CHECK:           %[[IGRAD:.*]] = torch_c.from_builtin_tensor %[[CONV_COLLAPSED]] : tensor<2x128x64x64xf32> -> !torch.vtensor<[2,128,64,64],f32>
  // CHECK:           %[[SUM_EMPTY:.*]] = tensor.empty() : tensor<16xf32>
  // CHECK:           %[[SUM_FILLED:.*]] = linalg.fill ins(%[[CST0]] : f32) outs(%[[SUM_EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
  // CHECK:           %[[SUM_GEN:.*]] = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>], iterator_types = ["reduction", "parallel", "reduction", "reduction"]} ins(%[[T0]] : tensor<2x16x33x33xf32>) outs(%[[SUM_FILLED]] : tensor<16xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN_B:.*]]: f32, %[[ACC_B:.*]]: f32):
  // CHECK-NEXT:        %[[B_RES:.*]] = arith.addf %[[IN_B]], %[[ACC_B]] : f32
  // CHECK-NEXT:        linalg.yield %[[B_RES]] : f32
  // CHECK-NEXT:      } -> tensor<16xf32>
  // CHECK:           %[[BIAS:.*]] = torch_c.from_builtin_tensor %[[SUM_GEN]] : tensor<16xf32> -> !torch.vtensor<[16],f32>
  // CHECK:           return %[[IGRAD]], %[[BIAS]] : !torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16],f32>
  %true = torch.constant.bool true
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int4 = torch.constant.int 4
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %true, %false, %true : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>
  %result0, %result1, %result2 = torch.aten.convolution_backward %arg0, %arg1, %arg2, %0, %1, %1, %1, %false, %2, %int4, %3 : !torch.vtensor<[2,16,33,33],f32>, !torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16,32,2,2],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int, !torch.list<bool> -> !torch.vtensor<[2,128,64,64],f32>, !torch.none, !torch.vtensor<[16],f32>
  return %result0, %result2 : !torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16],f32>
}

// -----

// CHECK-LABEL:   func.func @convolution_backward_weights_1x1s_0x0p_1x1d_1g(
// CHECK-SAME:                                                %[[VAL_0:.*]]: !torch.vtensor<[2,16,63,63],f32>, %[[VAL_1:.*]]: !torch.vtensor<[2,128,64,64],f32>,
// CHECK-SAME:                                                %[[VAL_2:.*]]: !torch.vtensor<[16,128,2,2],f32>,
// CHECK-SAME:                                                %[[VAL_3:.*]]: !torch.vtensor<[],f32>) -> (!torch.vtensor<[16,128,2,2],f32>, !torch.vtensor<[16],f32>) {
func.func @convolution_backward_weights_1x1s_0x0p_1x1d_1g(%arg0: !torch.vtensor<[2,16,63,63],f32>, %arg1: !torch.vtensor<[2,128,64,64],f32>, %arg2: !torch.vtensor<[16,128,2,2],f32>, %arg3: !torch.vtensor<[],f32>) -> (!torch.vtensor<[16,128,2,2],f32>, !torch.vtensor<[16],f32>) {
  // CHECK:           %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:           %[[T1:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[2,128,64,64],f32> -> tensor<2x128x64x64xf32>
  // CHECK:           %[[T0:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,16,63,63],f32> -> tensor<2x16x63x63xf32>
  // CHECK:           %[[OUT0_EMPTY:.*]] = tensor.empty() : tensor<16x128x2x2xf32>
  // CHECK:           %[[OUT0_FILLED:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[OUT0_EMPTY]] : tensor<16x128x2x2xf32>) -> tensor<16x128x2x2xf32>
  // CHECK:           %[[CONV:.*]] = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1, d2 + d5, d3 + d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d0, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%[[T1]], %[[T0]] : tensor<2x128x64x64xf32>, tensor<2x16x63x63xf32>) outs(%[[OUT0_FILLED]] : tensor<16x128x2x2xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN:.*]]: f32, %[[IN1:.*]]: f32, %[[OUT:.*]]: f32):
  // CHECK-NEXT:        %[[MUL:.*]] = arith.mulf %[[IN]], %[[IN1]] : f32
  // CHECK-NEXT:        %[[CONV_RES:.*]] = arith.addf %[[MUL]], %[[OUT]] : f32
  // CHECK-NEXT:        linalg.yield %[[CONV_RES]] : f32
  // CHECK-NEXT:      } -> tensor<16x128x2x2xf32>
  // CHECK:           %[[WGRAD:.*]] = torch_c.from_builtin_tensor %[[CONV]] : tensor<16x128x2x2xf32> -> !torch.vtensor<[16,128,2,2],f32>
  // CHECK:           %[[SUM_EMPTY:.*]] = tensor.empty() : tensor<16xf32>
  // CHECK:           %[[SUM_FILLED:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[SUM_EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
  // CHECK:           %[[SUM_GEN:.*]] = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>], iterator_types = ["reduction", "parallel", "reduction", "reduction"]} ins(%[[T0]] : tensor<2x16x63x63xf32>) outs(%[[SUM_FILLED]] : tensor<16xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN_B:.*]]: f32, %[[ACC_B:.*]]: f32):
  // CHECK-NEXT:        %[[B_RES:.*]] = arith.addf %[[IN_B]], %[[ACC_B]] : f32
  // CHECK-NEXT:        linalg.yield %[[B_RES]] : f32
  // CHECK-NEXT:      } -> tensor<16xf32>
  // CHECK:           %[[BIAS:.*]] = torch_c.from_builtin_tensor %[[SUM_GEN]] : tensor<16xf32> -> !torch.vtensor<[16],f32>
  // CHECK:           return %[[WGRAD]], %[[BIAS]] : !torch.vtensor<[16,128,2,2],f32>, !torch.vtensor<[16],f32>
  %true = torch.constant.bool true
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %false, %true, %true : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>
  %result0, %result1, %result2 = torch.aten.convolution_backward %arg0, %arg1, %arg2, %0, %1, %2, %1, %false, %2, %int1, %3 : !torch.vtensor<[2,16,63,63],f32>, !torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16,128,2,2],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int, !torch.list<bool> -> !torch.none, !torch.vtensor<[16,128,2,2],f32>, !torch.vtensor<[16],f32>
  return %result1, %result2 : !torch.vtensor<[16,128,2,2],f32>, !torch.vtensor<[16],f32>
}

// -----

// CHECK-LABEL:   func.func @convolution_backward_weights_2x2s_2x2p_2x2d_1g(
// CHECK-SAME:                                                %[[VAL_0:.*]]: !torch.vtensor<[2,32,33,33],f32>, %[[VAL_1:.*]]: !torch.vtensor<[2,128,64,64],f32>,
// CHECK-SAME:                                                %[[VAL_2:.*]]: !torch.vtensor<[32,128,2,2],f32>,
// CHECK-SAME:                                                %[[VAL_3:.*]]: !torch.vtensor<[],f32>) -> (!torch.vtensor<[32,128,2,2],f32>, !torch.vtensor<[32],f32>) {
func.func @convolution_backward_weights_2x2s_2x2p_2x2d_1g(%arg0: !torch.vtensor<[2,32,33,33],f32>, %arg1: !torch.vtensor<[2,128,64,64],f32>, %arg2: !torch.vtensor<[32,128,2,2],f32>, %arg3: !torch.vtensor<[],f32>) -> (!torch.vtensor<[32,128,2,2],f32>, !torch.vtensor<[32],f32>) {
  // CHECK:           %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:           %[[T1:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[2,128,64,64],f32> -> tensor<2x128x64x64xf32>
  // CHECK:           %[[T0:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,32,33,33],f32> -> tensor<2x32x33x33xf32>
  // CHECK:           %[[PAD:.*]] = tensor.pad %[[T1]] low[0, 0, 2, 2] high[0, 0, 2, 2]
  // CHECK-NEXT:      ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
  // CHECK-NEXT:      tensor.yield %[[CST]] : f32
  // CHECK-NEXT:      } : tensor<2x128x64x64xf32> to tensor<2x128x68x68xf32>
  // CHECK:           %[[OUT0_EMPTY:.*]] = tensor.empty() : tensor<32x128x2x2xf32>
  // CHECK:           %[[OUT0_FILLED:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[OUT0_EMPTY]] : tensor<32x128x2x2xf32>) -> tensor<32x128x2x2xf32>
  // CHECK:           %[[CONV:.*]] = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1, d2 * 2 + d5 * 2, d3 * 2 + d6 * 2)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d0, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%[[PAD]], %[[T0]] : tensor<2x128x68x68xf32>, tensor<2x32x33x33xf32>) outs(%[[OUT0_FILLED]] : tensor<32x128x2x2xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN:.*]]: f32, %[[IN1:.*]]: f32, %[[OUT:.*]]: f32):
  // CHECK-NEXT:        %[[MUL:.*]] = arith.mulf %[[IN]], %[[IN1]] : f32
  // CHECK-NEXT:        %[[CONV_RES:.*]] = arith.addf %[[MUL]], %[[OUT]] : f32
  // CHECK-NEXT:        linalg.yield %[[CONV_RES]] : f32
  // CHECK-NEXT:      } -> tensor<32x128x2x2xf32>
  // CHECK:           %[[WGRAD:.*]] = torch_c.from_builtin_tensor %[[CONV]] : tensor<32x128x2x2xf32> -> !torch.vtensor<[32,128,2,2],f32>
  // CHECK:           %[[SUM_EMPTY:.*]] = tensor.empty() : tensor<32xf32>
  // CHECK:           %[[SUM_FILLED:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[SUM_EMPTY]] : tensor<32xf32>) -> tensor<32xf32>
  // CHECK:           %[[SUM_GEN:.*]] = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>], iterator_types = ["reduction", "parallel", "reduction", "reduction"]} ins(%[[T0]] : tensor<2x32x33x33xf32>) outs(%[[SUM_FILLED]] : tensor<32xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN_B:.*]]: f32, %[[ACC_B:.*]]: f32):
  // CHECK-NEXT:        %[[B_RES:.*]] = arith.addf %[[IN_B]], %[[ACC_B]] : f32
  // CHECK-NEXT:        linalg.yield %[[B_RES]] : f32
  // CHECK-NEXT:      } -> tensor<32xf32>
  // CHECK:           %[[BIAS:.*]] = torch_c.from_builtin_tensor %[[SUM_GEN]] : tensor<32xf32> -> !torch.vtensor<[32],f32>
  // CHECK:           return %[[WGRAD]], %[[BIAS]] : !torch.vtensor<[32,128,2,2],f32>, !torch.vtensor<[32],f32>
  %true = torch.constant.bool true
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %false, %true, %true : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>
  %result0, %result1, %result2 = torch.aten.convolution_backward %arg0, %arg1, %arg2, %0, %1, %1, %1, %false, %2, %int1, %3 : !torch.vtensor<[2,32,33,33],f32>, !torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[32,128,2,2],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int, !torch.list<bool> -> !torch.none, !torch.vtensor<[32,128,2,2],f32>, !torch.vtensor<[32],f32>
  return %result1, %result2 : !torch.vtensor<[32,128,2,2],f32>, !torch.vtensor<[32],f32>
}

// -----

// CHECK-LABEL:   func.func @convolution_backward_weights_2x2s_2x2p_2x2d_4g(
// CHECK-SAME:                                                %[[VAL_0:.*]]: !torch.vtensor<[2,16,33,33],f32>, %[[VAL_1:.*]]: !torch.vtensor<[2,128,64,64],f32>,
// CHECK-SAME:                                                %[[VAL_2:.*]]: !torch.vtensor<[16,32,2,2],f32>,
// CHECK-SAME:                                                %[[VAL_3:.*]]: !torch.vtensor<[],f32>) -> (!torch.vtensor<[16,32,2,2],f32>, !torch.vtensor<[16],f32>) {
func.func @convolution_backward_weights_2x2s_2x2p_2x2d_4g(%arg0: !torch.vtensor<[2,16,33,33],f32>, %arg1: !torch.vtensor<[2,128,64,64],f32>, %arg2: !torch.vtensor<[16,32,2,2],f32>, %arg3: !torch.vtensor<[],f32>) -> (!torch.vtensor<[16,32,2,2],f32>, !torch.vtensor<[16],f32>) {
  // CHECK:           %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:           %[[T1:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[2,128,64,64],f32> -> tensor<2x128x64x64xf32>
  // CHECK:           %[[T0:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,16,33,33],f32> -> tensor<2x16x33x33xf32>
  // CHECK:           %[[T0_EXP:.*]] = tensor.expand_shape %[[T0]] {{\[\[0\], \[1, 2\], \[3\], \[4\]\]}} output_shape [2, 4, 4, 33, 33] : tensor<2x16x33x33xf32> into tensor<2x4x4x33x33xf32>
  // CHECK:           %[[T1_EXP:.*]] = tensor.expand_shape %[[T1]] {{\[\[0\], \[1, 2\], \[3\], \[4\]\]}} output_shape [2, 4, 32, 64, 64] : tensor<2x128x64x64xf32> into tensor<2x4x32x64x64xf32>
  // CHECK:           %[[PAD:.*]] = tensor.pad %[[T1_EXP]] low[0, 0, 0, 2, 2] high[0, 0, 0, 2, 2]
  // CHECK-NEXT:      ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
  // CHECK-NEXT:        tensor.yield %[[CST]] : f32
  // CHECK-NEXT:      } : tensor<2x4x32x64x64xf32> to tensor<2x4x32x68x68xf32>
  // CHECK:           %[[OUT0_EMPTY:.*]] = tensor.empty() : tensor<4x4x32x2x2xf32>
  // CHECK:           %[[OUT0_FILLED:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[OUT0_EMPTY]] : tensor<4x4x32x2x2xf32>) -> tensor<4x4x32x2x2xf32>
  // CHECK:           %[[CONV:.*]] = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d0, d2, d3 * 2 + d6 * 2, d4 * 2 + d7 * 2)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d0, d1, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%[[PAD]], %[[T0_EXP]] : tensor<2x4x32x68x68xf32>, tensor<2x4x4x33x33xf32>) outs(%[[OUT0_FILLED]] : tensor<4x4x32x2x2xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN:.*]]: f32, %[[IN1:.*]]: f32, %[[OUT:.*]]: f32):
  // CHECK-NEXT:        %[[MUL:.*]] = arith.mulf %[[IN]], %[[IN1]] : f32
  // CHECK-NEXT:        %[[CONV_RES:.*]] = arith.addf %[[MUL]], %[[OUT]] : f32
  // CHECK-NEXT:        linalg.yield %[[CONV_RES]] : f32
  // CHECK-NEXT:      } -> tensor<4x4x32x2x2xf32>
  // CHECK:           %[[COLLAPSED:.*]] = tensor.collapse_shape %[[CONV]] {{\[\[0, 1\], \[2\], \[3\], \[4\]\]}} : tensor<4x4x32x2x2xf32> into tensor<16x32x2x2xf32>
  // CHECK:           %[[WGRAD:.*]] = torch_c.from_builtin_tensor %[[COLLAPSED]] : tensor<16x32x2x2xf32> -> !torch.vtensor<[16,32,2,2],f32>
  // CHECK:           %[[SUM_EMPTY:.*]] = tensor.empty() : tensor<16xf32>
  // CHECK:           %[[SUM_FILLED:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[SUM_EMPTY]] : tensor<16xf32>) -> tensor<16xf32>
  // CHECK:           %[[SUM_GEN:.*]] = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>], iterator_types = ["reduction", "parallel", "reduction", "reduction"]} ins(%[[T0]] : tensor<2x16x33x33xf32>) outs(%[[SUM_FILLED]] : tensor<16xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN_B:.*]]: f32, %[[ACC_B:.*]]: f32):
  // CHECK-NEXT:        %[[B_RES:.*]] = arith.addf %[[IN_B]], %[[ACC_B]] : f32
  // CHECK-NEXT:        linalg.yield %[[B_RES]] : f32
  // CHECK-NEXT:      } -> tensor<16xf32>
  // CHECK:           %[[BIAS:.*]] = torch_c.from_builtin_tensor %[[SUM_GEN]] : tensor<16xf32> -> !torch.vtensor<[16],f32>
  // CHECK:           return %[[WGRAD]], %[[BIAS]] : !torch.vtensor<[16,32,2,2],f32>, !torch.vtensor<[16],f32>
  %true = torch.constant.bool true
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int4 = torch.constant.int 4
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %false, %true, %true : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>
  %result0, %result1, %result2 = torch.aten.convolution_backward %arg0, %arg1, %arg2, %0, %1, %1, %1, %false, %2, %int4, %3 : !torch.vtensor<[2,16,33,33],f32>, !torch.vtensor<[2,128,64,64],f32>, !torch.vtensor<[16,32,2,2],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int, !torch.list<bool> -> !torch.none, !torch.vtensor<[16,32,2,2],f32>, !torch.vtensor<[16],f32>
  return %result1, %result2 : !torch.vtensor<[16,32,2,2],f32>, !torch.vtensor<[16],f32>
}

// -----

// CHECK-LABEL:   func.func @convolution_backward_input_1x1x1s_1x0x1p_1x1x1d_1g(
// CHECK-SAME:                                                %[[VAL_0:.*]]: !torch.vtensor<[1,4,64,64,64],f32>, %[[VAL_1:.*]]: !torch.vtensor<[1,320,64,64,64],f32>,
// CHECK-SAME:                                                %[[VAL_2:.*]]: !torch.vtensor<[4,320,3,1,3],f32>,
// CHECK-SAME:                                                %[[VAL_3:.*]]: !torch.vtensor<[],f32>) -> (!torch.vtensor<[1,320,64,64,64],f32>, !torch.vtensor<[4],f32>) {
func.func @convolution_backward_input_1x1x1s_1x0x1p_1x1x1d_1g(%arg0: !torch.vtensor<[1,4,64,64,64],f32>, %arg1: !torch.vtensor<[1,320,64,64,64],f32>, %arg2: !torch.vtensor<[4,320,3,1,3],f32>, %arg3: !torch.vtensor<[],f32>) -> (!torch.vtensor<[1,320,64,64,64],f32>, !torch.vtensor<[4],f32>) {
  // CHECK:           %[[CST0:.*]] = arith.constant 0 : index
  // CHECK:           %[[CST2:.*]] = arith.constant 2 : index
  // CHECK:           %[[CST0F:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:           %[[WT:.*]] = torch_c.to_builtin_tensor %[[VAL_2]] : !torch.vtensor<[4,320,3,1,3],f32> -> tensor<4x320x3x1x3xf32>
  // CHECK:           %[[GO:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[1,4,64,64,64],f32> -> tensor<1x4x64x64x64xf32>
  // CHECK:           %[[W_EMPTY:.*]] = tensor.empty() : tensor<4x320x3x1x3xf32>
  // CHECK:           %[[W_FILLED:.*]] = linalg.fill ins(%[[CST0F]] : f32) outs(%[[W_EMPTY]] : tensor<4x320x3x1x3xf32>) -> tensor<4x320x3x1x3xf32>
  // CHECK:           %[[W_REV:.*]] = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[WT]] : tensor<4x320x3x1x3xf32>) outs(%[[W_FILLED]] : tensor<4x320x3x1x3xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN_W:.*]]: f32, %[[OUT_W:.*]]: f32):
  // CHECK-NEXT:        %[[I0:.*]] = linalg.index 0 : index
  // CHECK-NEXT:        %[[I1:.*]] = linalg.index 1 : index
  // CHECK-NEXT:        %[[I2:.*]] = linalg.index 2 : index
  // CHECK-NEXT:        %[[I4:.*]] = linalg.index 4 : index
  // CHECK-NEXT:        %[[R2:.*]] = arith.subi %[[CST2]], %[[I2]] : index
  // CHECK-NEXT:        %[[R4:.*]] = arith.subi %[[CST2]], %[[I4]] : index
  // CHECK-NEXT:        %[[EX:.*]] = tensor.extract %[[WT]][%[[I0]], %[[I1]], %[[R2]], %[[CST0]], %[[R4]]] : tensor<4x320x3x1x3xf32>
  // CHECK-NEXT:        linalg.yield %[[EX]] : f32
  // CHECK-NEXT:      } -> tensor<4x320x3x1x3xf32>
  // CHECK:           %[[PAD:.*]] = tensor.pad %[[GO]] low[0, 0, 1, 0, 1] high[0, 0, 1, 0, 1]
  // CHECK:           ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
  // CHECK:             tensor.yield %[[CST0F]] : f32
  // CHECK:           } : tensor<1x4x64x64x64xf32> to tensor<1x4x66x64x66xf32>
  // CHECK:           %[[OUT_EMPTY:.*]] = tensor.empty() : tensor<1x320x64x64x64xf32>
  // CHECK:           %[[OUT_FILLED:.*]] = linalg.fill ins(%[[CST0F]] : f32) outs(%[[OUT_EMPTY]] : tensor<1x320x64x64x64xf32>) -> tensor<1x320x64x64x64xf32>
  // CHECK:           %[[CONV:.*]] = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d4 + d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d5, d1, d6, d7, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PAD]], %[[W_REV]] : tensor<1x4x66x64x66xf32>, tensor<4x320x3x1x3xf32>) outs(%[[OUT_FILLED]] : tensor<1x320x64x64x64xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN:.*]]: f32, %[[IN1:.*]]: f32, %[[OUT:.*]]: f32):
  // CHECK-NEXT:        %[[MUL:.*]] = arith.mulf %[[IN]], %[[IN1]] : f32
  // CHECK-NEXT:        %[[ACC:.*]] = arith.addf %[[MUL]], %[[OUT]] : f32
  // CHECK-NEXT:        linalg.yield %[[ACC]] : f32
  // CHECK-NEXT:      } -> tensor<1x320x64x64x64xf32>
  // CHECK:           %[[IGRAD:.*]] = torch_c.from_builtin_tensor %[[CONV]] : tensor<1x320x64x64x64xf32> -> !torch.vtensor<[1,320,64,64,64],f32>
  // CHECK:           %[[SUM_EMPTY:.*]] = tensor.empty() : tensor<4xf32>
  // CHECK:           %[[SUM_FILLED:.*]] = linalg.fill ins(%[[CST0F]] : f32) outs(%[[SUM_EMPTY]] : tensor<4xf32>) -> tensor<4xf32>
  // CHECK:           %[[SUM_GEN:.*]] = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d1)>], iterator_types = ["reduction", "parallel", "reduction", "reduction", "reduction"]} ins(%[[GO]] : tensor<1x4x64x64x64xf32>) outs(%[[SUM_FILLED]] : tensor<4xf32>) {
  // CHECK-NEXT:      ^bb0(%[[IN_B:.*]]: f32, %[[ACC_B:.*]]: f32):
  // CHECK-NEXT:        %[[B_RES:.*]] = arith.addf %[[IN_B]], %[[ACC_B]] : f32
  // CHECK-NEXT:        linalg.yield %[[B_RES]] : f32
  // CHECK-NEXT:      } -> tensor<4xf32>
  // CHECK:           %[[BIAS:.*]] = torch_c.from_builtin_tensor %[[SUM_GEN]] : tensor<4xf32> -> !torch.vtensor<[4],f32>
  // CHECK:           return %[[IGRAD]], %[[BIAS]] : !torch.vtensor<[1,320,64,64,64],f32>, !torch.vtensor<[4],f32>
  %true = torch.constant.bool true
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int0, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int0, %int0, %int0 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct %true, %false, %true : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>
  %result0, %result1, %result2 = torch.aten.convolution_backward %arg0, %arg1, %arg2, %0, %1, %2, %1, %false, %3, %int1, %4 : !torch.vtensor<[1,4,64,64,64],f32>, !torch.vtensor<[1,320,64,64,64],f32>, !torch.vtensor<[4,320,3,1,3],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int, !torch.list<bool> -> !torch.vtensor<[1,320,64,64,64],f32>, !torch.none, !torch.vtensor<[4],f32>
  return %result0, %result2 : !torch.vtensor<[1,320,64,64,64],f32>, !torch.vtensor<[4],f32>
}

// -----
