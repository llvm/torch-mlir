// RUN: npcomp-opt -lower-to-npcomprt-abi -split-input-file -verify-diagnostics <%s | FileCheck %s --dump-input=fail

// CHECK:      npcomprt.module_metadata
// CHECK-NEXT:   npcomprt.func_metadata {funcName = @identity, numInputs = 1 : i32, numOutputs = 1 : i32}
// CHECK-NEXT:   npcomprt.func_metadata {funcName = @basic, numInputs = 1 : i32, numOutputs = 1 : i32}


// CHECK-LABEL:   func @identity(
// CHECK-SAME:                   %[[VAL_0:.*]]: !npcomprt.tensor) -> !npcomprt.tensor {
// CHECK:           return %[[VAL_0]] : !npcomprt.tensor
// CHECK:         }
func @identity(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  return %arg0 : tensor<?xf32>
}

// CHECK-LABEL:   func @basic(
// CHECK-SAME:                %[[VAL_0:.*]]: !npcomprt.tensor) -> !npcomprt.tensor {
// CHECK:           %[[VAL_1:.*]] = constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = npcomprt.get_extent %[[VAL_0]], %[[VAL_1]]
// CHECK:           %[[VAL_3:.*]] = shape.from_extents %[[VAL_2]]
// CHECK:           %[[VAL_4:.*]] = shape.to_extent_tensor %[[VAL_3]]
// CHECK:           %[[VAL_5:.*]] = tcp.alloc_memref %[[VAL_4]] : memref<?xf32>
// CHECK:           %[[VAL_6:.*]] = npcomprt.to_memref %[[VAL_0]] : memref<*xf32>
// CHECK:           %[[VAL_7:.*]] = memref_cast %[[VAL_6]] : memref<*xf32> to memref<?xf32>
// CHECK:           linalg.copy(%[[VAL_7]], %[[VAL_5]]) : memref<?xf32>, memref<?xf32>
// CHECK:           %[[VAL_8:.*]] = memref_cast %[[VAL_5]] : memref<?xf32> to memref<*xf32>
// CHECK:           %[[VAL_9:.*]] = npcomprt.from_memref %[[VAL_8]] : memref<*xf32>
// CHECK:           return %[[VAL_9]] : !npcomprt.tensor
// CHECK:         }

func @basic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %shape = shape.shape_of %arg0 : tensor<?xf32> -> tensor<?xindex>
  %memref = tcp.alloc_memref %shape : memref<?xf32>
  tensor_store %arg0, %memref : memref<?xf32>
  %ret = tensor_load %memref : memref<?xf32>
  return %ret: tensor<?xf32>
}

// -----


// CHECK: npcomprt.global @g dense<7.000000e+00> : tensor<10xf32>
tcp.global @g dense<7.0> : tensor<10xf32>
// CHECK-LABEL: func @gets_global() -> !npcomprt.tensor
func @gets_global() -> tensor<10xf32> {
// CHECK:    %[[GMEMREF:.*]] = npcomprt.get_global @g : memref<*xf32>
// CHECK:    %[[ORIGMEMREF:.*]] = memref_cast %[[GMEMREF]] : memref<*xf32> to memref<10xf32>
// CHECK:    %[[RETMEMREF:.*]] = memref_cast %[[ORIGMEMREF:.*]] : memref<10xf32> to memref<*xf32>
// CHECK:    %[[RET:.*]] = npcomprt.from_memref %[[RETMEMREF]] : memref<*xf32>
// CHECK:    return %[[RET]] : !npcomprt.tensor
  %0 = tcp.get_global_memref @g : memref<10xf32>
  %1 = tensor_load %0 : memref<10xf32>
  return %1 : tensor<10xf32>
}

// -----

// expected-error @+1 {{func not expressible with npcomprt ABI}}
func @unhandled_abi_type_on_public_func(%arg0: i32) {
  return
}
