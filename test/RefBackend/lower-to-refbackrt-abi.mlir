// RUN: npcomp-opt -lower-to-refbackrt-abi -split-input-file -verify-diagnostics <%s | FileCheck %s --dump-input=fail

// Test module metadata.

// CHECK:      refbackrt.module_metadata
// CHECK-NEXT:   refbackrt.func_metadata {funcName = @f_2inputs_0outputs, numInputs = 2 : i32, numOutputs = 0 : i32}
// CHECK-NEXT:   refbackrt.func_metadata {funcName = @f_1input_2outputs, numInputs = 1 : i32, numOutputs = 2 : i32}

// This function only exists to test its metadata above.
func @f_2inputs_0outputs(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
  return
}

// This function only exists to test its metadata above.
func @f_1input_2outputs(%arg0: memref<?xf32>) -> (memref<?xf32>, memref<?xf32>) {
  return %arg0, %arg0 : memref<?xf32>, memref<?xf32>
}

// -----

// Test ABI conversions.

// CHECK-LABEL:   func @identity(%arg0: memref<*xf32>) -> memref<*xf32>
func @identity(%arg0: memref<?xf32>) -> memref<?xf32> {
  // CHECK: return %arg0
  return %arg0 : memref<?xf32>
}


// -----

// CHECK-LABEL: func @use_of_arg(%arg0: memref<*xf32>)
func @use_of_arg(%arg0: memref<?xf32>) {
  // CHECK-NEXT: %[[MEMREF:.*]] = memref_cast %arg0 : memref<*xf32> to memref<?xf32>
  %c0 = constant 0 : index
  %0 = dim %arg0, %c0 : memref<?xf32>
  // CHECK-NEXT: %[[C0:.*]] = constant 0 : index
  // CHECK-NEXT: dim %[[MEMREF]], %[[C0]] : memref<?xf32>
  return
}

// -----

// CHECK-LABEL: func @multiple_blocks(%arg0: memref<*xf32>) -> memref<*xf32>
func @multiple_blocks(%arg0: memref<?xf32>) -> memref<?xf32> {
  // CHECK-NEXT:   %[[INMEMREF:.*]] = memref_cast %arg0 : memref<*xf32> to memref<?xf32>
  // CHECK-NEXT:   br ^bb1(%[[INMEMREF]] : memref<?xf32>)
  br ^bb1(%arg0: memref<?xf32>)
  // CHECK-NEXT: ^bb1(%[[BBARG:.*]]: memref<?xf32>):
^bb1(%bbarg: memref<?xf32>):
  // CHECK-NEXT:   %[[OUTMEMREF:.*]] = memref_cast %[[BBARG]] : memref<?xf32> to memref<*xf32>
  // CHECK-NEXT:   return %[[OUTMEMREF]] : memref<*xf32>
  return %bbarg : memref<?xf32>
}

// -----

// Test diagnostics.

// expected-error @+1 {{func not expressible with refbackrt ABI}}
func @unhandled_abi_type_on_public_func(%arg0: i32) {
  return
}
