// RUN: npcomp-opt -refback-lower-to-llvm -split-input-file <%s | FileCheck %s --dump-input=fail

// Test input/output arg marshaling.

// CHECK-LABEL:   llvm.func @__refbackrt_wrapper_inputs1results2(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: !llvm.ptr<ptr<i8>>,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: !llvm.ptr<ptr<i8>>) {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_2]]] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_5:.*]] = llvm.bitcast %[[VAL_4]] : !llvm.ptr<i8> to !llvm.ptr<i64>
// CHECK:           %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_8:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_7]]] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_10:.*]] = llvm.bitcast %[[VAL_9]] : !llvm.ptr<i8> to !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_11:.*]] = llvm.load %[[VAL_10]] : !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_12:.*]] = llvm.call @inputs1results2(%[[VAL_6]], %[[VAL_11]]) : (i64, !llvm.ptr<i8>) -> !llvm.struct<(struct<(i64, ptr<i8>)>, struct<(i64, ptr<i8>)>)>
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_14:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_13]]] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_15:.*]] = llvm.load %[[VAL_14]] : !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_16:.*]] = llvm.bitcast %[[VAL_15]] : !llvm.ptr<i8> to !llvm.ptr<struct<(i64, ptr<i8>)>>
// CHECK:           %[[VAL_17:.*]] = llvm.extractvalue %[[VAL_12]][0 : i32] : !llvm.struct<(struct<(i64, ptr<i8>)>, struct<(i64, ptr<i8>)>)>
// CHECK:           llvm.store %[[VAL_17]], %[[VAL_16]] : !llvm.ptr<struct<(i64, ptr<i8>)>>
// CHECK:           %[[VAL_18:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_19:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_18]]] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_20:.*]] = llvm.load %[[VAL_19]] : !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_21:.*]] = llvm.bitcast %[[VAL_20]] : !llvm.ptr<i8> to !llvm.ptr<struct<(i64, ptr<i8>)>>
// CHECK:           %[[VAL_22:.*]] = llvm.extractvalue %[[VAL_12]][1 : i32] : !llvm.struct<(struct<(i64, ptr<i8>)>, struct<(i64, ptr<i8>)>)>
// CHECK:           llvm.store %[[VAL_22]], %[[VAL_21]] : !llvm.ptr<struct<(i64, ptr<i8>)>>
// CHECK:           llvm.return
// CHECK:         }

// CHECK-LABEL:   llvm.func @__refbackrt_wrapper_inputs1results1(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: !llvm.ptr<ptr<i8>>,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: !llvm.ptr<ptr<i8>>) {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_2]]] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_5:.*]] = llvm.bitcast %[[VAL_4]] : !llvm.ptr<i8> to !llvm.ptr<i64>
// CHECK:           %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_8:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_7]]] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_10:.*]] = llvm.bitcast %[[VAL_9]] : !llvm.ptr<i8> to !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_11:.*]] = llvm.load %[[VAL_10]] : !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_12:.*]] = llvm.call @inputs1results1(%[[VAL_6]], %[[VAL_11]]) : (i64, !llvm.ptr<i8>) -> !llvm.struct<(i64, ptr<i8>)>
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_14:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_13]]] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_15:.*]] = llvm.load %[[VAL_14]] : !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_16:.*]] = llvm.bitcast %[[VAL_15]] : !llvm.ptr<i8> to !llvm.ptr<struct<(i64, ptr<i8>)>>
// CHECK:           llvm.store %[[VAL_12]], %[[VAL_16]] : !llvm.ptr<struct<(i64, ptr<i8>)>>
// CHECK:           llvm.return
// CHECK:         }

/// CHECK-LABEL:   llvm.func @__refbackrt_wrapper_inputs1results0(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: !llvm.ptr<ptr<i8>>,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: !llvm.ptr<ptr<i8>>) {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_3:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_2]]] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_5:.*]] = llvm.bitcast %[[VAL_4]] : !llvm.ptr<i8> to !llvm.ptr<i64>
// CHECK:           %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_8:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_7]]] : (!llvm.ptr<ptr<i8>>, i32) -> !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_9:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_10:.*]] = llvm.bitcast %[[VAL_9]] : !llvm.ptr<i8> to !llvm.ptr<ptr<i8>>
// CHECK:           %[[VAL_11:.*]] = llvm.load %[[VAL_10]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.call @inputs1results0(%[[VAL_6]], %[[VAL_11]]) : (i64, !llvm.ptr<i8>) -> ()
// CHECK:           llvm.return
// CHECK:         }
// CHECK:         llvm.func @__npcomp_compiler_rt_abort_if(i1, !llvm.ptr<i8>)
// CHECK:         llvm.mlir.global internal constant @__npcomp_internal_constant_inputs1results0("inputs1results0")
// CHECK:         llvm.mlir.global internal constant @__npcomp_internal_constant_inputs1results1("inputs1results1")
// CHECK:         llvm.mlir.global internal constant @__npcomp_internal_constant_inputs1results2("inputs1results2")

// CHECK-LABEL:   llvm.mlir.global internal constant @__npcomp_func_descriptors() : !llvm.array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(15 : i32) : i32
// CHECK:           %[[VAL_3:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_0]][0 : i32, 0 : i32] : !llvm.array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.addressof @__npcomp_internal_constant_inputs1results0 : !llvm.ptr<array<15 x i8>>
// CHECK:           %[[VAL_5:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_1]], %[[VAL_1]]] : (!llvm.ptr<array<15 x i8>>, i32, i32) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_6:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_3]][0 : i32, 1 : i32] : !llvm.array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.addressof @__refbackrt_wrapper_inputs1results0 : !llvm.ptr<func<void (ptr<ptr<i8>>, ptr<ptr<i8>>)>>
// CHECK:           %[[VAL_8:.*]] = llvm.bitcast %[[VAL_7]] : !llvm.ptr<func<void (ptr<ptr<i8>>, ptr<ptr<i8>>)>> to !llvm.ptr<i8>
// CHECK:           %[[VAL_9:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_6]][0 : i32, 2 : i32] : !llvm.array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>
// CHECK:           %[[VAL_10:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_11:.*]] = llvm.insertvalue %[[VAL_10]], %[[VAL_9]][0 : i32, 3 : i32] : !llvm.array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_12]], %[[VAL_11]][0 : i32, 4 : i32] : !llvm.array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>
// CHECK:           %[[VAL_14:.*]] = llvm.mlir.constant(15 : i32) : i32
// CHECK:           %[[VAL_15:.*]] = llvm.insertvalue %[[VAL_14]], %[[VAL_13]][1 : i32, 0 : i32] : !llvm.array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>
// CHECK:           %[[VAL_16:.*]] = llvm.mlir.addressof @__npcomp_internal_constant_inputs1results1 : !llvm.ptr<array<15 x i8>>
// CHECK:           %[[VAL_17:.*]] = llvm.getelementptr %[[VAL_16]]{{\[}}%[[VAL_1]], %[[VAL_1]]] : (!llvm.ptr<array<15 x i8>>, i32, i32) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_18:.*]] = llvm.insertvalue %[[VAL_17]], %[[VAL_15]][1 : i32, 1 : i32] : !llvm.array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>
// CHECK:           %[[VAL_19:.*]] = llvm.mlir.addressof @__refbackrt_wrapper_inputs1results1 : !llvm.ptr<func<void (ptr<ptr<i8>>, ptr<ptr<i8>>)>>
// CHECK:           %[[VAL_20:.*]] = llvm.bitcast %[[VAL_19]] : !llvm.ptr<func<void (ptr<ptr<i8>>, ptr<ptr<i8>>)>> to !llvm.ptr<i8>
// CHECK:           %[[VAL_21:.*]] = llvm.insertvalue %[[VAL_20]], %[[VAL_18]][1 : i32, 2 : i32] : !llvm.array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>
// CHECK:           %[[VAL_22:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_23:.*]] = llvm.insertvalue %[[VAL_22]], %[[VAL_21]][1 : i32, 3 : i32] : !llvm.array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>
// CHECK:           %[[VAL_24:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_25:.*]] = llvm.insertvalue %[[VAL_24]], %[[VAL_23]][1 : i32, 4 : i32] : !llvm.array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>
// CHECK:           %[[VAL_26:.*]] = llvm.mlir.constant(15 : i32) : i32
// CHECK:           %[[VAL_27:.*]] = llvm.insertvalue %[[VAL_26]], %[[VAL_25]][2 : i32, 0 : i32] : !llvm.array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>
// CHECK:           %[[VAL_28:.*]] = llvm.mlir.addressof @__npcomp_internal_constant_inputs1results2 : !llvm.ptr<array<15 x i8>>
// CHECK:           %[[VAL_29:.*]] = llvm.getelementptr %[[VAL_28]]{{\[}}%[[VAL_1]], %[[VAL_1]]] : (!llvm.ptr<array<15 x i8>>, i32, i32) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_30:.*]] = llvm.insertvalue %[[VAL_29]], %[[VAL_27]][2 : i32, 1 : i32] : !llvm.array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>
// CHECK:           %[[VAL_31:.*]] = llvm.mlir.addressof @__refbackrt_wrapper_inputs1results2 : !llvm.ptr<func<void (ptr<ptr<i8>>, ptr<ptr<i8>>)>>
// CHECK:           %[[VAL_32:.*]] = llvm.bitcast %[[VAL_31]] : !llvm.ptr<func<void (ptr<ptr<i8>>, ptr<ptr<i8>>)>> to !llvm.ptr<i8>
// CHECK:           %[[VAL_33:.*]] = llvm.insertvalue %[[VAL_32]], %[[VAL_30]][2 : i32, 2 : i32] : !llvm.array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>
// CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_35:.*]] = llvm.insertvalue %[[VAL_34]], %[[VAL_33]][2 : i32, 3 : i32] : !llvm.array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>
// CHECK:           %[[VAL_36:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_37:.*]] = llvm.insertvalue %[[VAL_36]], %[[VAL_35]][2 : i32, 4 : i32] : !llvm.array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>
// CHECK:           llvm.return %[[VAL_37]] : !llvm.array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>
// CHECK:         }

// CHECK-LABEL:   llvm.mlir.global external constant @_mlir___npcomp_module_descriptor() : !llvm.struct<(i32, ptr<struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>)> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.struct<(i32, ptr<struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>)>
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_0]][0 : i32] : !llvm.struct<(i32, ptr<struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>)>
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.addressof @__npcomp_func_descriptors : !llvm.ptr<array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>>
// CHECK:           %[[VAL_4:.*]] = llvm.bitcast %[[VAL_3]] : !llvm.ptr<array<3 x struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>> to !llvm.ptr<struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_2]][1 : i32] : !llvm.struct<(i32, ptr<struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>)>
// CHECK:           llvm.return %[[VAL_5]] : !llvm.struct<(i32, ptr<struct<(i32, ptr<i8>, ptr<i8>, i32, i32)>>)>
// CHECK:         }

refbackrt.module_metadata {
  refbackrt.func_metadata {funcName = @inputs1results0, numInputs = 1 : i32, numOutputs = 0 : i32}
  refbackrt.func_metadata {funcName = @inputs1results1, numInputs = 1 : i32, numOutputs = 1 : i32}
  refbackrt.func_metadata {funcName = @inputs1results2, numInputs = 1 : i32, numOutputs = 2 : i32}
}

func @inputs1results0(%arg0: memref<*xf32>) {
  return
}

func @inputs1results1(%arg0: memref<*xf32>) -> memref<*xf32> {
  return %arg0 : memref<*xf32>
}

func @inputs1results2(%arg0: memref<*xf32>) -> (memref<*xf32>, memref<*xf32>) {
  return %arg0, %arg0 : memref<*xf32>, memref<*xf32>
}

// -----

// Test emission of compiler runtime functions.

// CHECK:         llvm.mlir.global internal constant @[[STRSYM:.*]]("msg\00")
// CHECK:         llvm.func @__npcomp_compiler_rt_abort_if(i1, !llvm.ptr<i8>)

// CHECK-LABEL:   llvm.func @calls_abort_if(
// CHECK-SAME:                              %[[VAL_0:.*]]: i1) {
// CHECK:         %[[VAL_0:.*]] = llvm.mlir.addressof @[[STRSYM]] : !llvm.ptr<array<4 x i8>>
// CHECK:         %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[VAL_2:.*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]], %[[VAL_1]]] : (!llvm.ptr<array<4 x i8>>, i32, i32) -> !llvm.ptr<i8>
// CHECK:         llvm.call @__npcomp_compiler_rt_abort_if(%[[VAL_3:.*]], %[[VAL_2]]) : (i1, !llvm.ptr<i8>) -> ()
// CHECK:         llvm.return

func @calls_abort_if(%arg0: i1) {
  refbackrt.abort_if %arg0, "msg"
  return
}
