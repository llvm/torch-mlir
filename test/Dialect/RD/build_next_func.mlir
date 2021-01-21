// RUN: npcomp-opt -rd-build-next-func %s | FileCheck --dump-input=fail %s

// CHECK: rd.pipeline_def
"rd.pipeline_def"() ( {
  // CHECK: @definition
  func @definition(%start: i64, %end: i64) -> !rd.Dataset {
    %0 = rd.range %start to %end by %start : (i64, i64, i64) -> !rd.Dataset
    %1 = rd.inline_map @double(%0) : (!rd.Dataset) -> !rd.Dataset
    %2 = rd.filter %1 excluding @less_than_five : (!rd.Dataset) -> !rd.Dataset
    return %2 : !rd.Dataset
  }
  // CHECK: func @next(%[[ITR:.*]]: !rd.Iterator) -> !rd.Dataset {
  // CHECK: %[[RANGE_PTR:.*]] = rd.iterator_index %[[ITR]] [0]
  // CHECK: %[[RANGE_DS:.*]] = "rd.range.next"(%[[RANGE_PTR]])
  // CHECK: %[[MAP_PTR:.*]] = rd.iterator_index %[[ITR]] [1]
  // CHECK: %[[MAP_DS:.*]] = "rd.inline_map.next"(%[[RANGE_DS]], %[[MAP_PTR]])
  // CHECK: %[[FILTER_PTR:.*]] = rd.iterator_index %[[ITR]] [2]
  // CHECK: %[[FILTER_DS:.*]] = "rd.filter.next"(%[[MAP_DS]], %[[FILTER_PTR]])
  // CHECK: return %[[FILTER_DS]]
  // CHECK: }
  // CHECK: rd.pipeline_def_terminator
  rd.pipeline_def_terminator
}) { sym_name = "range_map_filter"} : () -> ()
