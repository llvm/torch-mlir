// RUN: npcomp-opt <%s | npcomp-opt | FileCheck %s --dump-input=fail

// CHECK: npcomprt.module_metadata
npcomprt.module_metadata {
  // CHECK: npcomprt.func_metadata
  npcomprt.func_metadata {funcName = @f, numInputs = 1 : i32, numOutputs = 0 : i32}
}

// CHECK-LABEL: func @f
// CHECK-SAME: !npcomprt.tensor
func @f(%arg0: !npcomprt.tensor) {
  return
}

// CHECK-LABEL: npcomprt.global @g dense<0.0{{.*}}> : tensor<10xf32>
npcomprt.global @g dense<0.0> : tensor<10xf32>
func @uses_global() {
  npcomprt.get_global @g : memref<*xf32>
  return
}