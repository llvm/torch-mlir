// RUN: npcomp-opt -split-input-file %s | npcomp-opt -canonicalize | FileCheck --dump-input=fail %s

// CHECK-LABEL: func @simple_range
func @simple_range(%start: i64, %end: i64) -> !rd.Dataset {
  // CHECK: %[[DS:.*]] = rd.range
  %0 = rd.range %start to %end  : (i64, i64) -> !rd.Dataset
  // CHECK: return %[[DS]]
  return %0 : !rd.Dataset
}
