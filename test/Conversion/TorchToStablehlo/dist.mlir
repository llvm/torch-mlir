// RUN: torch-mlir-opt <%s -convert-torch-to-stablehlo -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL:   func.func @torch.aten._cdist_forward$p3(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !torch.vtensor<[2,5,6,8],f32>,
// CHECK-SAME:                                            %[[VAL_1:.*]]: !torch.vtensor<[2,5,3,8],f32>) -> !torch.vtensor<[2,5,3,6],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[2,5,6,8],f32> -> tensor<2x5x6x8xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[2,5,3,8],f32> -> tensor<2x5x3x8xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.float 3.000000e+00
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_6:.*]] = stablehlo.reshape %[[VAL_3]] : (tensor<2x5x3x8xf32>) -> tensor<2x5x3x1x8xf32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.reshape %[[VAL_2]] : (tensor<2x5x6x8xf32>) -> tensor<2x5x1x6x8xf32>
// CHECK:           %[[VAL_8:.*]] = chlo.broadcast_subtract %[[VAL_6]], %[[VAL_7]] : (tensor<2x5x3x1x8xf32>, tensor<2x5x1x6x8xf32>) -> tensor<2x5x3x6x8xf32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.abs %[[VAL_8]] : tensor<2x5x3x6x8xf32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.constant dense<3.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_11:.*]] = chlo.broadcast_power %[[VAL_9]], %[[VAL_10]] : (tensor<2x5x3x6x8xf32>, tensor<f32>) -> tensor<2x5x3x6x8xf32>
// CHECK:           %[[VAL_12:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_13:.*]] = stablehlo.reduce(%[[VAL_11]] init: %[[VAL_12]]) applies stablehlo.add across dimensions = [4] : (tensor<2x5x3x6x8xf32>, tensor<f32>) -> tensor<2x5x3x6xf32>
// CHECK:           %[[VAL_14:.*]] = stablehlo.constant dense<0.333333343> : tensor<f32>
// CHECK:           %[[VAL_15:.*]] = chlo.broadcast_power %[[VAL_13]], %[[VAL_14]] : (tensor<2x5x3x6xf32>, tensor<f32>) -> tensor<2x5x3x6xf32>
// CHECK:           %[[VAL_16:.*]] = torch_c.from_builtin_tensor %[[VAL_15]] : tensor<2x5x3x6xf32> -> !torch.vtensor<[2,5,3,6],f32>
// CHECK:           return %[[VAL_16]] : !torch.vtensor<[2,5,3,6],f32>
// CHECK:         }
func.func @torch.aten._cdist_forward$p3(%arg0: !torch.vtensor<[2,5,6,8],f32>, %arg1: !torch.vtensor<[2,5,3,8],f32>) -> !torch.vtensor<[2,5,3,6],f32> {
    %float3.000000e00 = torch.constant.float 3.000000e+00
    %int1 = torch.constant.int 1
    %0 = torch.aten._cdist_forward %arg1, %arg0, %float3.000000e00, %int1 : !torch.vtensor<[2,5,3,8],f32>, !torch.vtensor<[2,5,6,8],f32>, !torch.float, !torch.int -> !torch.vtensor<[2,5,3,6],f32>
    return %0 : !torch.vtensor<[2,5,3,6],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten._cdist_forward$use_euclidean_dist(
// CHECK-SAME:                                                            %[[VAL_0:.*]]: !torch.vtensor<[5,10,8],f32>,
// CHECK-SAME:                                                            %[[VAL_1:.*]]: !torch.vtensor<[5,28,8],f32>) -> !torch.vtensor<[5,28,10],f32> {
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 5
// CHECK:           %[[VAL_3:.*]] = torch.constant.int 28
// CHECK:           %[[VAL_4:.*]] = torch.constant.int 8
// CHECK:           %[[VAL_5:.*]] = torch.prim.ListConstruct %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_6:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_7:.*]] = torch.aten.expand %[[VAL_1]], %[[VAL_5]], %[[VAL_6]] : !torch.vtensor<[5,28,8],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,28,8],f32>
// CHECK:           %[[VAL_8:.*]] = torch_c.to_builtin_tensor %[[VAL_7]] : !torch.vtensor<[5,28,8],f32> -> tensor<5x28x8xf32>
// CHECK:           %[[VAL_9:.*]] = torch.constant.int 5
// CHECK:           %[[VAL_10:.*]] = torch.constant.int 28
// CHECK:           %[[VAL_11:.*]] = torch.constant.int 8
// CHECK:           %[[VAL_12:.*]] = torch.prim.ListConstruct %[[VAL_9]], %[[VAL_10]], %[[VAL_11]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_13:.*]] = torch_c.to_i64 %[[VAL_9]]
// CHECK:           %[[VAL_14:.*]] = torch_c.to_i64 %[[VAL_10]]
// CHECK:           %[[VAL_15:.*]] = torch_c.to_i64 %[[VAL_11]]
// CHECK:           %[[VAL_16:.*]] = shape.shape_of %[[VAL_8]] : tensor<5x28x8xf32> -> tensor<3xindex>
// CHECK:           %[[VAL_17:.*]] = shape.num_elements %[[VAL_16]] : tensor<3xindex> -> index
// CHECK:           %[[VAL_18:.*]] = arith.index_cast %[[VAL_17]] : index to i64
// CHECK:           %[[VAL_19:.*]] = tensor.from_elements %[[VAL_13]], %[[VAL_14]], %[[VAL_15]] : tensor<3xi64>
// CHECK:           %[[VAL_20:.*]] = stablehlo.dynamic_reshape %[[VAL_8]], %[[VAL_19]] : (tensor<5x28x8xf32>, tensor<3xi64>) -> tensor<5x28x8xf32>
// CHECK:           %[[VAL_21:.*]] = torch.constant.int 5
// CHECK:           %[[VAL_22:.*]] = torch.constant.int 10
// CHECK:           %[[VAL_23:.*]] = torch.constant.int 8
// CHECK:           %[[VAL_24:.*]] = torch.prim.ListConstruct %[[VAL_21]], %[[VAL_22]], %[[VAL_23]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_25:.*]] = torch.constant.bool false
// CHECK:           %[[VAL_26:.*]] = torch.aten.expand %[[VAL_0]], %[[VAL_24]], %[[VAL_25]] : !torch.vtensor<[5,10,8],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,10,8],f32>
// CHECK:           %[[VAL_27:.*]] = torch_c.to_builtin_tensor %[[VAL_26]] : !torch.vtensor<[5,10,8],f32> -> tensor<5x10x8xf32>
// CHECK:           %[[VAL_28:.*]] = torch.constant.int 5
// CHECK:           %[[VAL_29:.*]] = torch.constant.int 10
// CHECK:           %[[VAL_30:.*]] = torch.constant.int 8
// CHECK:           %[[VAL_31:.*]] = torch.prim.ListConstruct %[[VAL_28]], %[[VAL_29]], %[[VAL_30]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_32:.*]] = torch_c.to_i64 %[[VAL_28]]
// CHECK:           %[[VAL_33:.*]] = torch_c.to_i64 %[[VAL_29]]
// CHECK:           %[[VAL_34:.*]] = torch_c.to_i64 %[[VAL_30]]
// CHECK:           %[[VAL_35:.*]] = shape.shape_of %[[VAL_27]] : tensor<5x10x8xf32> -> tensor<3xindex>
// CHECK:           %[[VAL_36:.*]] = shape.num_elements %[[VAL_35]] : tensor<3xindex> -> index
// CHECK:           %[[VAL_37:.*]] = arith.index_cast %[[VAL_36]] : index to i64
// CHECK:           %[[VAL_38:.*]] = tensor.from_elements %[[VAL_32]], %[[VAL_33]], %[[VAL_34]] : tensor<3xi64>
// CHECK:           %[[VAL_39:.*]] = stablehlo.dynamic_reshape %[[VAL_27]], %[[VAL_38]] : (tensor<5x10x8xf32>, tensor<3xi64>) -> tensor<5x10x8xf32>
// CHECK:           %[[VAL_40:.*]] = stablehlo.multiply %[[VAL_20]], %[[VAL_20]] : tensor<5x28x8xf32>
// CHECK:           %[[VAL_41:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_42:.*]] = stablehlo.reduce(%[[VAL_40]] init: %[[VAL_41]]) applies stablehlo.add across dimensions = [2] : (tensor<5x28x8xf32>, tensor<f32>) -> tensor<5x28xf32>
// CHECK:           %[[VAL_43:.*]] = stablehlo.multiply %[[VAL_39]], %[[VAL_39]] : tensor<5x10x8xf32>
// CHECK:           %[[VAL_44:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_45:.*]] = stablehlo.reduce(%[[VAL_43]] init: %[[VAL_44]]) applies stablehlo.add across dimensions = [2] : (tensor<5x10x8xf32>, tensor<f32>) -> tensor<5x10xf32>
// CHECK:           %[[VAL_46:.*]] = stablehlo.transpose %[[VAL_39]], dims = [0, 2, 1] : (tensor<5x10x8xf32>) -> tensor<5x8x10xf32>
// CHECK:           %[[VAL_47:.*]] = stablehlo.dot_general %[[VAL_20]], %[[VAL_46]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<5x28x8xf32>, tensor<5x8x10xf32>) -> tensor<5x28x10xf32>
// CHECK:           %[[VAL_48:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_49:.*]] = chlo.broadcast_multiply %[[VAL_47]], %[[VAL_48]] : (tensor<5x28x10xf32>, tensor<f32>) -> tensor<5x28x10xf32>
// CHECK:           %[[VAL_50:.*]] = stablehlo.reshape %[[VAL_42]] : (tensor<5x28xf32>) -> tensor<5x28x1xf32>
// CHECK:           %[[VAL_51:.*]] = stablehlo.reshape %[[VAL_45]] : (tensor<5x10xf32>) -> tensor<5x1x10xf32>
// CHECK:           %[[VAL_52:.*]] = chlo.broadcast_add %[[VAL_50]], %[[VAL_51]] : (tensor<5x28x1xf32>, tensor<5x1x10xf32>) -> tensor<5x28x10xf32>
// CHECK:           %[[VAL_53:.*]] = stablehlo.subtract %[[VAL_52]], %[[VAL_49]] : tensor<5x28x10xf32>
// CHECK:           %[[VAL_54:.*]] = stablehlo.sqrt %[[VAL_53]] : tensor<5x28x10xf32>
// CHECK:           %[[VAL_55:.*]] = torch.constant.int 5
// CHECK:           %[[VAL_56:.*]] = torch.constant.int 28
// CHECK:           %[[VAL_57:.*]] = torch.constant.int 10
// CHECK:           %[[VAL_58:.*]] = torch.prim.ListConstruct %[[VAL_55]], %[[VAL_56]], %[[VAL_57]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[VAL_59:.*]] = torch_c.to_i64 %[[VAL_55]]
// CHECK:           %[[VAL_60:.*]] = torch_c.to_i64 %[[VAL_56]]
// CHECK:           %[[VAL_61:.*]] = torch_c.to_i64 %[[VAL_57]]
// CHECK:           %[[VAL_62:.*]] = shape.shape_of %[[VAL_54]] : tensor<5x28x10xf32> -> tensor<3xindex>
// CHECK:           %[[VAL_63:.*]] = shape.num_elements %[[VAL_62]] : tensor<3xindex> -> index
// CHECK:           %[[VAL_64:.*]] = arith.index_cast %[[VAL_63]] : index to i64
// CHECK:           %[[VAL_65:.*]] = tensor.from_elements %[[VAL_59]], %[[VAL_60]], %[[VAL_61]] : tensor<3xi64>
// CHECK:           %[[VAL_66:.*]] = stablehlo.dynamic_reshape %[[VAL_54]], %[[VAL_65]] : (tensor<5x28x10xf32>, tensor<3xi64>) -> tensor<5x28x10xf32>
// CHECK:           %[[VAL_67:.*]] = torch_c.from_builtin_tensor %[[VAL_66]] : tensor<5x28x10xf32> -> !torch.vtensor<[5,28,10],f32>
// CHECK:           return %[[VAL_67]] : !torch.vtensor<[5,28,10],f32>
// CHECK:         }
func.func @torch.aten._cdist_forward$use_euclidean_dist(%arg0: !torch.vtensor<[5,10,8],f32>, %arg1: !torch.vtensor<[5,28,8],f32>) -> !torch.vtensor<[5,28,10],f32> {
    %int5 = torch.constant.int 5
    %int28 = torch.constant.int 28
    %int8 = torch.constant.int 8
    %0 = torch.prim.ListConstruct %int5, %int28, %int8 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.expand %arg1, %0, %false : !torch.vtensor<[5,28,8],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,28,8],f32>
    %int5_0 = torch.constant.int 5
    %int28_1 = torch.constant.int 28
    %int8_2 = torch.constant.int 8
    %2 = torch.prim.ListConstruct %int5_0, %int28_1, %int8_2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.view %1, %2 : !torch.vtensor<[5,28,8],f32>, !torch.list<int> -> !torch.vtensor<[5,28,8],f32>
    %int5_3 = torch.constant.int 5
    %int10 = torch.constant.int 10
    %int8_4 = torch.constant.int 8
    %4 = torch.prim.ListConstruct %int5_3, %int10, %int8_4 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_5 = torch.constant.bool false
    %5 = torch.aten.expand %arg0, %4, %false_5 : !torch.vtensor<[5,10,8],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,10,8],f32>
    %int5_6 = torch.constant.int 5
    %int10_7 = torch.constant.int 10
    %int8_8 = torch.constant.int 8
    %6 = torch.prim.ListConstruct %int5_6, %int10_7, %int8_8 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.aten.view %5, %6 : !torch.vtensor<[5,10,8],f32>, !torch.list<int> -> !torch.vtensor<[5,10,8],f32>
    %8 = torch.aten._euclidean_dist %3, %7 : !torch.vtensor<[5,28,8],f32>, !torch.vtensor<[5,10,8],f32> -> !torch.vtensor<[5,28,10],f32>
    %int5_9 = torch.constant.int 5
    %int28_10 = torch.constant.int 28
    %int10_11 = torch.constant.int 10
    %9 = torch.prim.ListConstruct %int5_9, %int28_10, %int10_11 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %10 = torch.aten.view %8, %9 : !torch.vtensor<[5,28,10],f32>, !torch.list<int> -> !torch.vtensor<[5,28,10],f32>
    return %10 : !torch.vtensor<[5,28,10],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten._cdist_forward$p2_no_euclid_dist(
// CHECK-SAME:                                                           %[[VAL_0:.*]]: !torch.vtensor<[4,5,3,16],f32>,
// CHECK-SAME:                                                           %[[VAL_1:.*]]: !torch.vtensor<[4,5,9,16],f32>) -> !torch.vtensor<[4,5,9,3],f32> {
// CHECK:           %[[VAL_2:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,5,3,16],f32> -> tensor<4x5x3x16xf32>
// CHECK:           %[[VAL_3:.*]] = torch_c.to_builtin_tensor %[[VAL_1]] : !torch.vtensor<[4,5,9,16],f32> -> tensor<4x5x9x16xf32>
// CHECK:           %[[VAL_4:.*]] = torch.constant.float 2.000000e+00
// CHECK:           %[[VAL_5:.*]] = torch.constant.int 2
// CHECK:           %[[VAL_6:.*]] = stablehlo.reshape %[[VAL_3]] : (tensor<4x5x9x16xf32>) -> tensor<4x5x9x1x16xf32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.reshape %[[VAL_2]] : (tensor<4x5x3x16xf32>) -> tensor<4x5x1x3x16xf32>
// CHECK:           %[[VAL_8:.*]] = chlo.broadcast_subtract %[[VAL_6]], %[[VAL_7]] : (tensor<4x5x9x1x16xf32>, tensor<4x5x1x3x16xf32>) -> tensor<4x5x9x3x16xf32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.multiply %[[VAL_8]], %[[VAL_8]] : tensor<4x5x9x3x16xf32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_11:.*]] = stablehlo.reduce(%[[VAL_9]] init: %[[VAL_10]]) applies stablehlo.add across dimensions = [4] : (tensor<4x5x9x3x16xf32>, tensor<f32>) -> tensor<4x5x9x3xf32>
// CHECK:           %[[VAL_12:.*]] = stablehlo.sqrt %[[VAL_11]] : tensor<4x5x9x3xf32>
// CHECK:           %[[VAL_13:.*]] = torch_c.from_builtin_tensor %[[VAL_12]] : tensor<4x5x9x3xf32> -> !torch.vtensor<[4,5,9,3],f32>
// CHECK:           return %[[VAL_13]] : !torch.vtensor<[4,5,9,3],f32>
// CHECK:         }
func.func @torch.aten._cdist_forward$p2_no_euclid_dist(%arg0: !torch.vtensor<[4,5,3,16],f32>, %arg1: !torch.vtensor<[4,5,9,16],f32>) -> !torch.vtensor<[4,5,9,3],f32> {
    %float2.000000e00 = torch.constant.float 2.000000e+00
    %int2 = torch.constant.int 2
    %0 = torch.aten._cdist_forward %arg1, %arg0, %float2.000000e00, %int2 : !torch.vtensor<[4,5,9,16],f32>, !torch.vtensor<[4,5,3,16],f32>, !torch.float, !torch.int -> !torch.vtensor<[4,5,9,3],f32>
    return %0 : !torch.vtensor<[4,5,9,3],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten._pdist_forward$p5(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !torch.vtensor<[6,10],f32>) -> !torch.vtensor<[15],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[6,10],f32> -> tensor<6x10xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 5.000000e+00
// CHECK:           %[[VAL_3:.*]] = stablehlo.reshape %[[VAL_1]] : (tensor<6x10xf32>) -> tensor<6x1x10xf32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.reshape %[[VAL_1]] : (tensor<6x10xf32>) -> tensor<1x6x10xf32>
// CHECK:           %[[VAL_5:.*]] = chlo.broadcast_subtract %[[VAL_3]], %[[VAL_4]] : (tensor<6x1x10xf32>, tensor<1x6x10xf32>) -> tensor<6x6x10xf32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.abs %[[VAL_5]] : tensor<6x6x10xf32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.constant dense<5.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_8:.*]] = chlo.broadcast_power %[[VAL_6]], %[[VAL_7]] : (tensor<6x6x10xf32>, tensor<f32>) -> tensor<6x6x10xf32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.reduce(%[[VAL_8]] init: %[[VAL_9]]) applies stablehlo.add across dimensions = [2] : (tensor<6x6x10xf32>, tensor<f32>) -> tensor<6x6xf32>
// CHECK:           %[[VAL_11:.*]] = stablehlo.constant dense<2.000000e-01> : tensor<f32>
// CHECK:           %[[VAL_12:.*]] = chlo.broadcast_power %[[VAL_10]], %[[VAL_11]] : (tensor<6x6xf32>, tensor<f32>) -> tensor<6x6xf32>
// CHECK:           %[[VAL_13:.*]] = stablehlo.reshape %[[VAL_12]] : (tensor<6x6xf32>) -> tensor<36xf32>
// CHECK:           %[[VAL_14:.*]] = stablehlo.constant dense<{{\[\[}}1], [2], [3], [4], [5], [8], [9], [10], [11], [15], [16], [17], [22], [23], [29]]> : tensor<15x1xi64>
// CHECK:           %[[VAL_15:.*]] = "stablehlo.gather"(%[[VAL_13]], %[[VAL_14]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<36xf32>, tensor<15x1xi64>) -> tensor<15xf32>
// CHECK:           %[[VAL_16:.*]] = torch_c.from_builtin_tensor %[[VAL_15]] : tensor<15xf32> -> !torch.vtensor<[15],f32>
// CHECK:           return %[[VAL_16]] : !torch.vtensor<[15],f32>
// CHECK:         }
func.func @torch.aten._pdist_forward$p5(%arg0: !torch.vtensor<[6,10],f32>) -> !torch.vtensor<[15],f32> {
    %float5.000000e00 = torch.constant.float 5.000000e+00
    %0 = torch.aten._pdist_forward %arg0, %float5.000000e00 : !torch.vtensor<[6,10],f32>, !torch.float -> !torch.vtensor<[15],f32>
    return %0 : !torch.vtensor<[15],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten._pdist_forward$p2(
// CHECK-SAME:                                            %[[VAL_0:.*]]: !torch.vtensor<[4,6],f32>) -> !torch.vtensor<[6],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[4,6],f32> -> tensor<4x6xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 2.000000e+00
// CHECK:           %[[VAL_3:.*]] = stablehlo.reshape %[[VAL_1]] : (tensor<4x6xf32>) -> tensor<4x1x6xf32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.reshape %[[VAL_1]] : (tensor<4x6xf32>) -> tensor<1x4x6xf32>
// CHECK:           %[[VAL_5:.*]] = chlo.broadcast_subtract %[[VAL_3]], %[[VAL_4]] : (tensor<4x1x6xf32>, tensor<1x4x6xf32>) -> tensor<4x4x6xf32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.multiply %[[VAL_5]], %[[VAL_5]] : tensor<4x4x6xf32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_8:.*]] = stablehlo.reduce(%[[VAL_6]] init: %[[VAL_7]]) applies stablehlo.add across dimensions = [2] : (tensor<4x4x6xf32>, tensor<f32>) -> tensor<4x4xf32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.sqrt %[[VAL_8]] : tensor<4x4xf32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.reshape %[[VAL_9]] : (tensor<4x4xf32>) -> tensor<16xf32>
// CHECK:           %[[VAL_11:.*]] = stablehlo.constant dense<{{\[\[}}1], [2], [3], [6], [7], [11]]> : tensor<6x1xi64>
// CHECK:           %[[VAL_12:.*]] = "stablehlo.gather"(%[[VAL_10]], %[[VAL_11]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<16xf32>, tensor<6x1xi64>) -> tensor<6xf32>
// CHECK:           %[[VAL_13:.*]] = torch_c.from_builtin_tensor %[[VAL_12]] : tensor<6xf32> -> !torch.vtensor<[6],f32>
// CHECK:           return %[[VAL_13]] : !torch.vtensor<[6],f32>
// CHECK:         }
func.func @torch.aten._pdist_forward$p2(%arg0: !torch.vtensor<[4,6],f32>) -> !torch.vtensor<[6],f32> {
    %float2.000000e00 = torch.constant.float 2.000000e+00
    %0 = torch.aten._pdist_forward %arg0, %float2.000000e00 : !torch.vtensor<[4,6],f32>, !torch.float -> !torch.vtensor<[6],f32>
    return %0 : !torch.vtensor<[6],f32>
}
