// RUN: npcomp-opt -e2e-lower-to-llvm <%s | FileCheck %s --dump-input=fail

// CHECK-LABEL: llvm.func @identity(%arg0: !llvm.i64, %arg1: !llvm<"i8*">) -> !llvm<"{ i64, i8* }">
func @identity(%arg0: memref<*xf32>) -> memref<*xf32> {
  return %arg0 : memref<*xf32>
}

// CHECK-LABEL: llvm.func @abort_if(
// CHECK-SAME:      %[[PRED:.*]]: !llvm.i1)
func @abort_if(%arg0: i1) {
  // CHECK: llvm.call @__npcomp_abort_if(%arg0) : (!llvm.i1) -> ()
  "tcp.abort_if"(%arg0) : (i1) -> ()
  return
}

