// RUN: npcomp-opt -e2e-lower-to-llvm -split-input-file <%s | FileCheck %s --dump-input=fail

// CHECK:         llvm.mlir.global internal constant @__npcomprt_global_data_buffer_g(dense<7.000000e+00> : tensor<3xf32>) : !llvm<"[3 x float]">
// CHECK:         llvm.mlir.global internal constant @__npcomprt_global_extents_g(dense<3> : tensor<1xi32>) : !llvm<"[1 x i32]">
// CHECK-LABEL:   llvm.mlir.global internal constant @g() : !llvm<"{ i32, i32*, i8* }"> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.undef : !llvm<"{ i32, i32*, i8* }">
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_0]][0 : i32] : !llvm<"{ i32, i32*, i8* }">
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.addressof @__npcomprt_global_extents_g : !llvm<"[1 x i32]*">
// CHECK:           %[[VAL_4:.*]] = llvm.bitcast %[[VAL_3]] : !llvm<"[1 x i32]*"> to !llvm<"i32*">
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_2]][1 : i32] : !llvm<"{ i32, i32*, i8* }">
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.addressof @__npcomprt_global_data_buffer_g : !llvm<"[3 x float]*">
// CHECK:           %[[VAL_7:.*]] = llvm.bitcast %[[VAL_6]] : !llvm<"[3 x float]*"> to !llvm<"i8*">
// CHECK:           %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_5]][2 : i32] : !llvm<"{ i32, i32*, i8* }">
// CHECK:           llvm.return %[[VAL_8]] : !llvm<"{ i32, i32*, i8* }">
// CHECK:         }
// CHECK-LABEL:   llvm.func @calls_get_global() -> !llvm<"{ i64, i8* }"> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.addressof @g : !llvm<"{ i32, i32*, i8* }*">
// CHECK:           %[[VAL_1:.*]] = llvm.call @__npcomp_compiler_rt_get_global(%[[VAL_0]]) : (!llvm<"{ i32, i32*, i8* }*">) -> !llvm<"{ i64, i8* }">
npcomprt.global @g dense<7.000000e+00> : tensor<3xf32>
func @calls_get_global() -> memref<*xf32> {
  %0 = npcomprt.get_global @g : memref<*xf32>
  return %0 : memref<*xf32>
}

// -----

// For scalars, we have to fake-up a size-1 data buffer array to make LLVM translation happy.
// CHECK: llvm.mlir.global internal constant @__npcomprt_global_data_buffer_g(dense<7.000000e+00> : tensor<f32>) : !llvm<"[1 x float]">     
// CHECK: llvm.mlir.global internal constant @__npcomprt_global_extents_g(dense<0> : tensor<1xi32>) : !llvm<"[1 x i32]">
npcomprt.global @g dense<7.0> : tensor<f32>

