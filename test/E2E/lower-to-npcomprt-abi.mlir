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
// CHECK:           %[[VAL_4:.*]] = tcp.alloc_memref %[[VAL_3]] : memref<?xf32>
// CHECK:           %[[VAL_5:.*]] = npcomprt.to_memref %[[VAL_0]] : memref<*xf32>
// CHECK:           %[[VAL_6:.*]] = memref_cast %[[VAL_5]] : memref<*xf32> to memref<?xf32>
// CHECK:           linalg.copy(%[[VAL_6]], %[[VAL_4]]) : memref<?xf32>, memref<?xf32>
// CHECK:           %[[VAL_7:.*]] = memref_cast %[[VAL_4]] : memref<?xf32> to memref<*xf32>
// CHECK:           %[[VAL_8:.*]] = npcomprt.from_memref %[[VAL_7]] : memref<*xf32>
// CHECK:           return %[[VAL_8]] : !npcomprt.tensor
// CHECK:         }

func @basic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %shape = shape.shape_of %arg0 : tensor<?xf32>
  %memref = tcp.alloc_memref %shape : memref<?xf32>
  tensor_store %arg0, %memref : memref<?xf32>
  %ret = tensor_load %memref : memref<?xf32>
  return %ret: tensor<?xf32>
}

// -----

// expected-error @+1 {{func not expressible with npcomprt ABI}}
func @unhandled_abi_type_on_public_func(%arg0: i32) {
  return
}
