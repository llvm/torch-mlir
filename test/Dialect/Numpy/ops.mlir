// RUN: npcomp-opt %s | npcomp-opt | FileCheck %s

// CHECK-LABEL: func @foo()
func @foo() -> i32 {
  %0 = constant 1 : i32
  // CHECK: %{{.*}} = numpy.foo %{{.*}} : i32
  %res = numpy.foo %0 : i32
  return %res : i32
}
