// RUN: npcomp-opt <%s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @f
// CHECK-SAME: !npcomp_rt.buffer_view
func @f(%arg0: !npcomp_rt.buffer_view) {
  return
}

