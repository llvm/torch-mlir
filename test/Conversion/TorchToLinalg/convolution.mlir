// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -canonicalize -split-input-file -mlir-print-local-scope -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func @torch.aten.convolution$nobias(
// CHECK:           %[[CONSTANT:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[FILL_RESULT:.*]] = linalg.fill ins(%[[CONSTANT]] : f32) outs(%{{.*}} : tensor<1x54x16x128x128xf32>) -> tensor<1x54x16x128x128xf32>
// CHECK:           %[[CONV3D:.*]] = linalg.conv_3d_ncdhw_fcdhw {{.*}} outs(%[[FILL_RESULT]] : tensor<1x54x16x128x128xf32>) -> tensor<1x54x16x128x128xf32>
func.func @torch.aten.convolution$nobias(%arg0: !torch.vtensor<[1,24,16,128,128],f16>, %arg1: !torch.vtensor<[54,24,1,1,1],f16>) -> !torch.vtensor<[1,54,16,128,128],f16> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %0 = torch.prim.ListConstruct %int0, %int0, %int0 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.aten.convolution %arg0, %arg1, %none, %2, %0, %1, %false, %3, %int1 : !torch.vtensor<[1,24,16,128,128],f16>, !torch.vtensor<[54,24,1,1,1],f16>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,54,16,128,128],f16>
  return %4 : !torch.vtensor<[1,54,16,128,128],f16>
}

// -----

// CHECK-LABEL: func.func @q_conv_test
// CHECK: %[[c3:.*]] = arith.constant 3 : i32
// CHECK: %[[c7:.*]] = arith.constant 7 : i32
// CHECK: %[[input:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[?,?,?,?],si8> -> tensor<?x?x?x?xi8>
// CHECK: %[[weight:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[?,?,?,?],si8> -> tensor<?x?x?x?xi8>
// CHECK: %[[conv:.*]] = linalg.conv_2d_nchw_fchw_q {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
// CHECK-SAME: ins(%[[input]], %[[weight]], %[[c7]], %[[c3]] : tensor<?x?x?x?xi8>, tensor<?x?x?x?xi8>, i32, i32)
// CHECK-SAME: outs(%[[convout:.*]] : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
func.func @q_conv_test(%arg0: !torch.vtensor<[?,?,?,?],si8>, %arg1: !torch.vtensor<[?,?,?,?],si8>, %arg2: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %float1.000000e-04 = torch.constant.float 1.000000e-04
  %int3 = torch.constant.int 3
  %int7 = torch.constant.int 7
  %float1.000000e-02 = torch.constant.float 1.000000e-02
  %int14 = torch.constant.int 14
  %0 = torch.aten.quantize_per_tensor %arg2, %float1.000000e-04, %int0, %int14 : !torch.vtensor<[?],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[?],!torch.qint32>
  %1 = torch.aten.dequantize.self %0 : !torch.vtensor<[?],!torch.qint32> -> !torch.vtensor<[?],f32>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
  %5 = torch.aten._make_per_tensor_quantized_tensor %arg0, %float1.000000e-02, %int7 : !torch.vtensor<[?,?,?,?],si8>, !torch.float, !torch.int -> !torch.vtensor<[?,?,?,?],!torch.qint8>
  %6 = torch.aten._make_per_tensor_quantized_tensor %arg1, %float1.000000e-02, %int3 : !torch.vtensor<[?,?,?,?],si8>, !torch.float, !torch.int -> !torch.vtensor<[?,?,?,?],!torch.qint8>
  %7 = torch.aten.quantize_per_tensor %1, %float1.000000e-04, %int0, %int14 : !torch.vtensor<[?],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[?],!torch.qint32>
  %8 = torch.aten.int_repr %7 : !torch.vtensor<[?],!torch.qint32> -> !torch.vtensor<[?],si32>
  %9 = torch.aten.convolution %5, %6, %8, %2, %3, %2, %false, %4, %int1 : !torch.vtensor<[?,?,?,?],!torch.qint8>, !torch.vtensor<[?,?,?,?],!torch.qint8>, !torch.vtensor<[?],si32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[?,?,?,?],si32>
  %10 = torch.aten._make_per_tensor_quantized_tensor %9, %float1.000000e-04, %int0 : !torch.vtensor<[?,?,?,?],si32>, !torch.float, !torch.int -> !torch.vtensor<[?,?,?,?],!torch.qint32>
  %11 = torch.aten.dequantize.tensor %10 : !torch.vtensor<[?,?,?,?],!torch.qint32> -> !torch.vtensor<[?,?,?,?],f32>
  return %11 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @conv_broadcast(
// CHECK-SAME:                              %[[arg0:.*]]: !torch.vtensor<[1,80,3000],f32>,
// CHECK-SAME:                              %[[arg1:.*]]: !torch.vtensor<[1024,80,3],f32>,
// CHECK-SAME:                              %[[arg2:.*]]: !torch.vtensor<[1024],f32>) -> !torch.vtensor<[1,1024,3000],f32> {
// CHECK:           %[[c0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       %[[input:.*]] = torch_c.to_builtin_tensor %[[arg0]] : !torch.vtensor<[1,80,3000],f32> -> tensor<1x80x3000xf32>
// CHECK-DAG:       %[[weight:.*]] = torch_c.to_builtin_tensor %[[arg1]] : !torch.vtensor<[1024,80,3],f32> -> tensor<1024x80x3xf32>
// CHECK-DAG:       %[[bias:.*]] = torch_c.to_builtin_tensor %[[arg2]] : !torch.vtensor<[1024],f32> -> tensor<1024xf32>
// CHECK:           %[[padInput:.*]] = tensor.pad %[[input]] low[0, 0, 1] high[0, 0, 1]
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<1x1024x3000xf32>
// CHECK:           %[[broadcastBias:.*]] = linalg.broadcast ins(%[[bias]] : tensor<1024xf32>) outs(%[[EMPTY]] : tensor<1x1024x3000xf32>) dimensions = [0, 2]
// CHECK:           %[[conv:.*]] = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>}
// CHECK-SAME:      ins(%[[padInput:.*]], %[[weight]] : tensor<1x80x3002xf32>, tensor<1024x80x3xf32>)
// CHECK-SAME:      outs(%[[broadcastBias]] : tensor<1x1024x3000xf32>) -> tensor<1x1024x3000xf32>
func.func @conv_broadcast(%arg0: !torch.vtensor<[1,80,3000],f32>, %arg1: !torch.vtensor<[1024,80,3],f32>, %arg2: !torch.vtensor<[1024],f32>) -> !torch.vtensor<[1,1024,3000],f32> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %1 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %2 = torch.aten.convolution %arg0, %arg1, %arg2, %0, %0, %0, %false, %1, %int1 : !torch.vtensor<[1,80,3000],f32>, !torch.vtensor<[1024,80,3],f32>, !torch.vtensor<[1024],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,1024,3000],f32>
  return %2 : !torch.vtensor<[1,1024,3000],f32>
}

// CHECK-LABEL:   func.func @transposedConv2D(
// CHECK-SAME:       %[[INPUT_TENSOR:.*]]: !torch.vtensor<[1,2,5,7],f32>) -> !torch.vtensor<[1,4,10,14],f32>
// CHECK:         = linalg.generic
// CHECK-SAME:       outs(%[[BROADCASTED_WEIGHTS_INIT:.*]] : tensor<4x2x3x3xf32>) {
// CHECK:         %[[WEIGHTS:.*]] = tensor.extract
// CHECK-SAME:       : tensor<2x4x3x3xf32>
// CHECK-NEXT:    linalg.yield %[[BROADCASTED_WEIGHTS:.*]] : f32
// CHECK-NEXT:    } -> tensor<4x2x3x3xf32>
// CHECK:         %[[BROADCASTED_BIAS:.*]] = linalg.broadcast ins(%[[BIAS:.*]] : tensor<4xf32>) outs(%[[BROADCASTED_BIAS_INIT:.*]] : tensor<1x4x11x15xf32>) dimensions = [0, 2, 3]
// CHECK:         %[[CONV_RESULT:.*]] = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
// CHECK-SAME:        ins(%[[INPUT_TENSOR_ADAPTED:.*]], %[[BROADCASTED_WEIGHTS:.*]] : tensor<1x2x13x17xf32>, tensor<4x2x3x3xf32>) outs(%[[BROADCASTED_BIAS:.*]] : tensor<1x4x11x15xf32>) -> tensor<1x4x11x15xf32>
// CHECK-NEXT:    %[[OUTPUT_TENSOR_DYN:.*]] = tensor.cast %[[CONV_RESULT:.*]] : tensor<1x4x11x15xf32> to tensor<1x4x?x?xf32>
// CHECK-NEXT:    %[[OUTPUT_TENSOR:.*]] = tensor.cast %[[OUTPUT_TENSOR_DYN:.*]] : tensor<1x4x?x?xf32> to tensor<1x4x10x14xf32>
func.func @transposedConv2D(%arg0: !torch.vtensor<[1,2,5,7],f32>) -> !torch.vtensor<[1,4,10,14],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %int0 = torch.constant.int 0
  %true = torch.constant.bool true
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_4_3_3_torch.float32> : tensor<2x4x3x3xf32>) : !torch.vtensor<[2,4,3,3],f32>
  %1 = torch.vtensor.literal(dense_resource<torch_tensor_4_torch.float32> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %2 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %5 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %6 = torch.aten.convolution %arg0, %0, %1, %2, %3, %4, %true, %5, %int1 : !torch.vtensor<[1,2,5,7],f32>, !torch.vtensor<[2,4,3,3],f32>, !torch.vtensor<[4],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,4,10,14],f32>
  return %6 : !torch.vtensor<[1,4,10,14],f32>
}

// CHECK-LABEL:   func.func @groupedConvolution2D(
// CHECK-SAME:       %[[INPUT_TENSOR:.*]]: !torch.vtensor<[1,4,5,7],f32>) -> !torch.vtensor<[1,4,5,7],f32>
// CHECK:         %[[BROADCASTED_BIAS:.*]] = linalg.broadcast ins(%[[BIAS:.*]] : tensor<4xf32>) outs(%[[BROADCASTED_BIAS_INIT:.*]] : tensor<1x4x5x7xf32>) dimensions = [0, 2, 3]
// CHECK:         %[[CONV_RESULT:.*]] = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
// CHECK-SAME:        ins(%[[INPUT_TENSOR_ADAPTED:.*]], %[[BROADCASTED_WEIGHTS:.*]] : tensor<1x2x2x7x9xf32>, tensor<2x2x2x3x3xf32>) outs(%[[BROADCASTED_BIAS:.*]] : tensor<1x2x2x5x7xf32>) -> tensor<1x2x2x5x7xf32>
// CHECK-NEXT:    %[[OUTPUT_TENSOR:.*]] = tensor.collapse_shape
// CHECK-SAME:        tensor<1x2x2x5x7xf32> into tensor<1x4x5x7xf32>
func.func @groupedConvolution2D(%arg0: !torch.vtensor<[1,4,5,7],f32>) -> !torch.vtensor<[1,4,5,7],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.vtensor.literal(dense_resource<torch_tensor_4_2_3_3_torch.float32> : tensor<4x2x3x3xf32>) : !torch.vtensor<[4,2,3,3],f32>
  %1 = torch.vtensor.literal(dense_resource<torch_tensor_4_torch.float32> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %5 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %6 = torch.aten.convolution %arg0, %0, %1, %2, %3, %4, %false, %5, %int2 : !torch.vtensor<[1,4,5,7],f32>, !torch.vtensor<[4,2,3,3],f32>, !torch.vtensor<[4],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,4,5,7],f32>
  return %6 : !torch.vtensor<[1,4,5,7],f32>
}

// CHECK-LABEL:   func.func @transposedGroupedConvolution2D(
// CHECK-SAME:       %[[INPUT_TENSOR:.*]]: !torch.vtensor<[1,2,5,7],f32>) -> !torch.vtensor<[1,4,10,14],f32>
// CHECK:         %[[BROADCASTED_BIAS:.*]] = linalg.broadcast ins(%[[BIAS:.*]] : tensor<4xf32>) outs(%[[BROADCASTED_BIAS_INIT:.*]] : tensor<1x4x11x15xf32>) dimensions = [0, 2, 3]
// CHECK:         %[[CONV_RESULT:.*]] = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
// CHECK-SAME:        ins(%[[INPUT_TENSOR_ADAPTED:.*]], %[[BROADCASTED_WEIGHTS:.*]] : tensor<1x2x1x13x17xf32>, tensor<2x2x1x3x3xf32>) outs(%[[BROADCASTED_BIAS:.*]] : tensor<1x2x2x11x15xf32>) -> tensor<1x2x2x11x15xf32>
// CHECK-NEXT:    %[[COLLAPSED_TENSOR:.*]] = tensor.collapse_shape
// CHECK-SAME:        tensor<1x2x2x11x15xf32> into tensor<1x4x11x15xf32>
// CHECK-NEXT:    %[[OUTPUT_TENSOR_DYN:.*]] = tensor.cast %[[COLLAPSED_TENSOR:.*]] : tensor<1x4x11x15xf32> to tensor<1x4x?x?xf32>
// CHECK-NEXT:    %[[OUTPUT_TENSOR:.*]] = tensor.cast %[[OUTPUT_TENSOR_DYN:.*]] : tensor<1x4x?x?xf32> to tensor<1x4x10x14xf32>
func.func @transposedGroupedConvolution2D(%arg0: !torch.vtensor<[1,2,5,7],f32>) -> !torch.vtensor<[1,4,10,14],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %int0 = torch.constant.int 0
  %true = torch.constant.bool true
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_2_3_3_torch.float32> : tensor<2x2x3x3xf32>) : !torch.vtensor<[2,2,3,3],f32>
  %1 = torch.vtensor.literal(dense_resource<torch_tensor_4_torch.float32> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  %2 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %5 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %6 = torch.aten.convolution %arg0, %0, %1, %2, %3, %4, %true, %5, %int2 : !torch.vtensor<[1,2,5,7],f32>, !torch.vtensor<[2,2,3,3],f32>, !torch.vtensor<[4],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,4,10,14],f32>
  return %6 : !torch.vtensor<[1,4,10,14],f32>
}

// CHECK-LABEL:   func.func @tranConv2dNegativePadding(
// CHECK-SAME:       %[[INPUT_VTENSOR:.*]]: !torch.vtensor<[1,1,4,7],f32>) -> !torch.vtensor<[1,2,6,3],f32>
// CHECK:            %[[IN_TENSOR:.*]] = torch_c.to_builtin_tensor %[[INPUT_VTENSOR:.*]] : !torch.vtensor<[1,1,4,7],f32> -> tensor<1x1x4x7xf32>
// CHECK:            %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[IN_TENSOR:.*]][0, 0, 0, 1] [1, 1, 4, 5] [1, 1, 1, 1] : tensor<1x1x4x7xf32> to tensor<1x1x4x5xf32>
// CHECK:            %[[INSERTED_SLICE:.*]] = tensor.insert_slice %[[EXTRACTED_SLICE:.*]] into %[[INIT_TENSOR:.*]][0, 0, 2, 0] [1, 1, 4, 5] [1, 1, 1, 1] : tensor<1x1x4x5xf32> into tensor<1x1x8x5xf32>
// CHECK:            %[[OUT_TENSOR:.*]] = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%[[INSERTED_SLICE:.*]], %[[WEIGHTS:.*]] : tensor<1x1x8x5xf32>, tensor<2x1x3x3xf32>) outs(%[[INIT_OUT_TENSOR:.*]] : tensor<1x2x6x3xf32>) -> tensor<1x2x6x3xf32>
// CHECK:            %[[OUT_VTENSOR:.*]] = torch_c.from_builtin_tensor %[[OUT_TENSOR:.*]] : tensor<1x2x6x3xf32> -> !torch.vtensor<[1,2,6,3],f32>
func.func @tranConv2dNegativePadding(%arg0: !torch.vtensor<[1, 1, 4, 7],f32>) -> !torch.vtensor<[1, 2, 6, 3],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %int0 = torch.constant.int 0
  %true = torch.constant.bool true
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_2_3_3_torch.float32> : tensor<1x2x3x3xf32>) : !torch.vtensor<[1,2,3,3],f32>
  %1 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.float32> : tensor<2xf32>) : !torch.vtensor<[2],f32>
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int0, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %5 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %6 = torch.aten.convolution %arg0, %0, %1, %2, %3, %4, %true, %5, %int1 : !torch.vtensor<[1, 1, 4, 7],f32>, !torch.vtensor<[1,2,3,3],f32>, !torch.vtensor<[2],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1, 2, 6, 3],f32>
  return %6 : !torch.vtensor<[1, 2, 6, 3],f32>
}
