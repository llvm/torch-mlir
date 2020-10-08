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

// CHECK-LABEL:   func @identity(%arg0: !refbackrt.tensor) -> !refbackrt.tensor
func @identity(%arg0: memref<?xf32>) -> memref<?xf32> {
  // The argument materialization.
  // In this test case, these go unused since, as described below, the new
  // argument value is seen immediately by the return op for some reason.
  // CHECK-NEXT: %[[INABIMEMREF:.*]] = refbackrt.to_memref %arg0 : memref<*xf32>
  // CHECK-NEXT: %[[MEMREF:.*]] = memref_cast %[[INABIMEMREF]] : memref<*xf32> to memref<?xf32>

  // TODO: Why do these target materializations not happen in this particular
  // test?
  // Somehow, the return op rewrite sees the new argument value immediately,
  // rather than the result of replaceUsesOfBlockArgument from
  // FuncOpSignatureConversion
  // Cxxxx-NEXT: %[[OUTABIMEMREF:.*]] = memref_cast %[[MEMREF]] : memref<?xf32> to memref<*xf32>
  // Cxxxx-NEXT: %[[RET:.*]] = refbackrt.from_memref %[[OUTABIMEMREF]] : memref<*xf32>
  // Cxxxx-NEXT: return %[[RET]]

  // CHECK-NEXT: return %arg0
  return %arg0 : memref<?xf32>
}


// -----

// CHECK-LABEL: func @use_of_arg(%arg0: !refbackrt.tensor)
func @use_of_arg(%arg0: memref<?xf32>) {
  // CHECK-NEXT: %[[INABIMEMREF:.*]] = refbackrt.to_memref %arg0 : memref<*xf32>
  // CHECK-NEXT: %[[MEMREF:.*]] = memref_cast %[[INABIMEMREF]] : memref<*xf32> to memref<?xf32>
  %c0 = constant 0 : index
  %0 = dim %arg0, %c0 : memref<?xf32>
  // CHECK-NEXT: %[[C0:.*]] = constant 0 : index
  // CHECK-NEXT: dim %[[MEMREF]], %[[C0]] : memref<?xf32>
  return
}

// -----

// CHECK-LABEL: func @multiple_blocks(%arg0: !refbackrt.tensor) -> !refbackrt.tensor
func @multiple_blocks(%arg0: memref<?xf32>) -> memref<?xf32> {
  // CHECK-NEXT:   %[[INABIMEMREF:.*]] = refbackrt.to_memref %arg0 : memref<*xf32>
  // CHECK-NEXT:   %[[INMEMREF:.*]] = memref_cast %[[INABIMEMREF]] : memref<*xf32> to memref<?xf32>
  // CHECK-NEXT:   br ^bb1(%[[INMEMREF]] : memref<?xf32>)
  br ^bb1(%arg0: memref<?xf32>)
  // CHECK-NEXT: ^bb1(%[[BBARG:.*]]: memref<?xf32>):
^bb1(%bbarg: memref<?xf32>):
  // CHECK-NEXT:   %[[OUTMEMREF:.*]] = memref_cast %[[BBARG]] : memref<?xf32> to memref<*xf32>
  // CHECK-NEXT:   %[[OUTABIMEMREF:.*]] = refbackrt.from_memref %[[OUTMEMREF]] : memref<*xf32>
  // CHECK-NEXT:   return %[[OUTABIMEMREF]] : !refbackrt.tensor
  return %bbarg : memref<?xf32>
}

// -----


// CHECK: refbackrt.global @g dense<7.000000e+00> : tensor<10xf32>
refback.global @g dense<7.0> : tensor<10xf32>
// CHECK-LABEL: func @gets_global() -> !refbackrt.tensor
func @gets_global() -> memref<10xf32> {
// CHECK:    %[[GMEMREF:.*]] = refbackrt.get_global @g : memref<*xf32>
// CHECK:    %[[ORIGMEMREF:.*]] = memref_cast %[[GMEMREF]] : memref<*xf32> to memref<10xf32>
// CHECK:    %[[OUTABIMEMREF:.*]] = memref_cast %[[ORIGMEMREF:.*]] : memref<10xf32> to memref<*xf32>
// CHECK:    %[[RET:.*]] = refbackrt.from_memref %[[OUTABIMEMREF]] : memref<*xf32>
// CHECK:    return %[[RET]] : !refbackrt.tensor
  %0 = refback.get_global_memref @g : memref<10xf32>
  return %0 : memref<10xf32>
}

// -----

// Test diagnostics.

// expected-error @+1 {{func not expressible with refbackrt ABI}}
func @unhandled_abi_type_on_public_func(%arg0: i32) {
  return
}
