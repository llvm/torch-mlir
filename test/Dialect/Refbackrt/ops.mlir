// RUN: npcomp-opt <%s | npcomp-opt | FileCheck %s --dump-input=fail

// CHECK: refbackrt.module_metadata
refbackrt.module_metadata {
  // CHECK: refbackrt.func_metadata
  refbackrt.func_metadata {funcName = @f, numInputs = 1 : i32, numOutputs = 0 : i32}
}

// CHECK-LABEL: func @f
// CHECK-SAME: !refbackrt.tensor
func @f(%arg0: !refbackrt.tensor) {
  return
}

// CHECK-LABEL: refbackrt.global @g dense<0.0{{.*}}> : tensor<10xf32>
refbackrt.global @g dense<0.0> : tensor<10xf32>
func @uses_global() {
  refbackrt.get_global @g : memref<*xf32>
  return
}
