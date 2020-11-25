// RUN: npcomp-opt <%s | npcomp-opt | FileCheck %s --dump-input=fail

// CHECK: refbackrt.module_metadata
refbackrt.module_metadata {
  // CHECK: refbackrt.func_metadata
  refbackrt.func_metadata {funcName = @f, numInputs = 1 : i32, numOutputs = 0 : i32}
}

func @f(%arg0: memref<*xf32>) {
  return
}
