// RUN: npcomp-opt -refback-lower-to-llvm -split-input-file <%s | FileCheck %s --dump-input=fail

// CHECK-LABEL:   llvm.func @malloc(!llvm.i64) -> !llvm.ptr<i8>
// CHECK:         llvm.func @__npcomp_compiler_rt_abort_if(!llvm.i1, !llvm.ptr<i8>)
// CHECK:         llvm.func @__npcomp_compiler_rt_to_memref(!llvm.ptr<i8>) -> !llvm.struct<(i64, ptr<i8>)>
// CHECK:         llvm.func @__npcomp_compiler_rt_from_memref(!llvm.i64, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK:         llvm.func @__npcomp_compiler_rt_get_global(!llvm.ptr<struct<(i32, ptr<i32>, ptr<i8>)>>) -> !llvm.struct<(i64, ptr<i8>)>
// CHECK:         llvm.mlir.global internal constant @__npcomprt_global_data_buffer_g(dense<7.000000e+00> : tensor<3xf32>) : !llvm.array<3 x float>
// CHECK:         llvm.mlir.global internal constant @__npcomprt_global_extents_g(dense<3> : tensor<1xi32>) : !llvm.array<1 x i32>

// CHECK-LABEL:   llvm.mlir.global internal constant @g() : !llvm.struct<(i32, ptr<i32>, ptr<i8>)> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.struct<(i32, ptr<i32>, ptr<i8>)>
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_0]][0 : i32] : !llvm.struct<(i32, ptr<i32>, ptr<i8>)>
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.addressof @__npcomprt_global_extents_g : !llvm.ptr<array<1 x i32>>
// CHECK:           %[[VAL_4:.*]] = llvm.bitcast %[[VAL_3]] : !llvm.ptr<array<1 x i32>> to !llvm.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_2]][1 : i32] : !llvm.struct<(i32, ptr<i32>, ptr<i8>)>
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.addressof @__npcomprt_global_data_buffer_g : !llvm.ptr<array<3 x float>>
// CHECK:           %[[VAL_7:.*]] = llvm.bitcast %[[VAL_6]] : !llvm.ptr<array<3 x float>> to !llvm.ptr<i8>
// CHECK:           %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_5]][2 : i32] : !llvm.struct<(i32, ptr<i32>, ptr<i8>)>
// CHECK:           llvm.return %[[VAL_8]] : !llvm.struct<(i32, ptr<i32>, ptr<i8>)>
// CHECK:         }

// CHECK-LABEL:   llvm.func @calls_get_global() -> !llvm.struct<(i64, ptr<i8>)> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.addressof @g : !llvm.ptr<struct<(i32, ptr<i32>, ptr<i8>)>>
// CHECK:           %[[VAL_1:.*]] = llvm.call @__npcomp_compiler_rt_get_global(%[[VAL_0]]) : (!llvm.ptr<struct<(i32, ptr<i32>, ptr<i8>)>>) -> !llvm.struct<(i64, ptr<i8>)>
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(2 : index) : !llvm.i64
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(8 : index) : !llvm.i64
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(8 : index) : !llvm.i64
// CHECK:           %[[VAL_6:.*]] = llvm.mul %[[VAL_3]], %[[VAL_4]] : !llvm.i64
// CHECK:           %[[VAL_7:.*]] = llvm.extractvalue %[[VAL_1]][0] : !llvm.struct<(i64, ptr<i8>)>
// CHECK:           %[[VAL_8:.*]] = llvm.mul %[[VAL_3]], %[[VAL_7]] : !llvm.i64
// CHECK:           %[[VAL_9:.*]] = llvm.add %[[VAL_8]], %[[VAL_2]] : !llvm.i64
// CHECK:           %[[VAL_10:.*]] = llvm.mul %[[VAL_9]], %[[VAL_5]] : !llvm.i64
// CHECK:           %[[VAL_11:.*]] = llvm.add %[[VAL_6]], %[[VAL_10]] : !llvm.i64
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.constant(false) : !llvm.i1
// CHECK:           %[[VAL_13:.*]] = llvm.call @malloc(%[[VAL_11]]) : (!llvm.i64) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_14:.*]] = llvm.extractvalue %[[VAL_1]][1] : !llvm.struct<(i64, ptr<i8>)>
// CHECK:           "llvm.intr.memcpy"(%[[VAL_13]], %[[VAL_14]], %[[VAL_11]], %[[VAL_12]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i1) -> ()
// CHECK:           %[[VAL_15:.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
// CHECK:           %[[VAL_16:.*]] = llvm.extractvalue %[[VAL_1]][0] : !llvm.struct<(i64, ptr<i8>)>
// CHECK:           %[[VAL_17:.*]] = llvm.insertvalue %[[VAL_16]], %[[VAL_15]][0] : !llvm.struct<(i64, ptr<i8>)>
// CHECK:           %[[VAL_18:.*]] = llvm.insertvalue %[[VAL_13]], %[[VAL_17]][1] : !llvm.struct<(i64, ptr<i8>)>
// CHECK:           llvm.return %[[VAL_18]] : !llvm.struct<(i64, ptr<i8>)>
// CHECK:         }
npcomprt.global @g dense<7.000000e+00> : tensor<3xf32>
func @calls_get_global() -> memref<*xf32> {
  %0 = npcomprt.get_global @g : memref<*xf32>
  return %0 : memref<*xf32>
}

// -----

// For scalars, we have to fake-up a size-1 data buffer array to make LLVM translation happy.
// CHECK: llvm.mlir.global internal constant @__npcomprt_global_data_buffer_g(dense<7.000000e+00> : tensor<f32>) : !llvm.array<1 x float>
// CHECK: llvm.mlir.global internal constant @__npcomprt_global_extents_g(dense<0> : tensor<1xi32>) : !llvm.array<1 x i32>
npcomprt.global @g dense<7.0> : tensor<f32>
