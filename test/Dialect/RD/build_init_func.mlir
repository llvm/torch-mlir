// RUN: npcomp-opt -rd-build-init-func %s | FileCheck --dump-input=fail %s

// CHECK: rd.pipeline_def
"rd.pipeline_def"() ( {
  // CHECK: @definition
  func @definition(%start: i64, %end: i64) -> !rd.Dataset {
    %0 = rd.range %start to %end by %start : (i64, i64, i64) -> !rd.Dataset
    %1 = rd.inline_map @double(%0) : (!rd.Dataset) -> !rd.Dataset
    %2 = rd.filter %1 excluding @less_than_five : (!rd.Dataset) -> !rd.Dataset
    return %2 : !rd.Dataset
  }
  // CHECK: llvm.func @init(%[[START:.*]]: !llvm.i64, %[[END:.*]]: !llvm.i64, %[[PTR:.*]]: !llvm.ptr<struct<
  // CHECK: %[[RANGE_PTR:.*]] = llvm.getelementptr
  // CHECK: %[[START_PTR:.*]] = llvm.getelementptr %[[RANGE_PTR]]
  // CHECK: llvm.store %[[START]], %[[START_PTR]]
  // CHECK: %[[END_PTR:.*]] = llvm.getelementptr %[[RANGE_PTR]]
  // CHECK: llvm.store %[[END]], %[[END_PTR]]
  // CHECK: %[[STEP_PTR:.*]] = llvm.getelementptr %[[RANGE_PTR]]
  // CHECK: llvm.store %[[START]], %[[STEP_PTR]]
  // CHECK: llvm.return
  // CHECK: rd.pipeline_def_terminator
  rd.pipeline_def_terminator
}) { sym_name = "range_map_filter"} : () -> ()
