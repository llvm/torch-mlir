// RUN: npcomp-opt -split-input-file %s | FileCheck --dump-input=fail %s

// CHECK-LABEL: func @simple_range
// CHECK-SAME:                    %[[START:.*]]: i64, %[[END:.*]]: i64
func @simple_range(%start: i64, %end: i64) -> !rd.Dataset {
  // CHECK: %[[DS:.*]] = rd.range %[[START]] to %[[END]]
  %0 = rd.range %start to %end by %start : (i64, i64, i64) -> !rd.Dataset
  // CHECK: return %[[DS]]
  return %0 : !rd.Dataset
}

// -----

// CHECK-LABEL: func @range_map_filter
// CHECK-SAME:                    %[[START:.*]]: i64, %[[END:.*]]: i64
func @range_map_filter(%start: i64, %end: i64) -> !rd.Dataset {
  // CHECK: %[[RANGE:.*]] = rd.range %[[START]] to %[[END]]
  %0 = rd.range %start to %end by %start : (i64, i64, i64) -> !rd.Dataset
  // CHECK: %[[DOUBLED:.*]] = rd.inline_map @double(%[[RANGE]])
  %1 = rd.inline_map @double(%0) : (!rd.Dataset) -> !rd.Dataset
  // CHECK: %[[FILTERED:.*]] = rd.filter %[[DOUBLED]] excluding @less_than_five
  %2 = rd.filter %1 excluding @less_than_five : (!rd.Dataset) -> !rd.Dataset
  // CHECK: return %[[FILTERED]]
  return %2 : !rd.Dataset
}

func @use_dataset() {
  %start = constant 0 : i64
  %end = constant 5 : i64
  %itr = rd.make_iterator @range_map_filter(%start, %end) : (i64, i64) -> !rd.Iterator
  br ^bb1

^bb1:
  %isValid, %value = rd.iterator_next %itr : (!rd.Iterator) -> (i1, i64)
  cond_br %isValid, ^bb2, ^bb3

^bb2:
  rd.print %value : i64
  br ^bb1

^bb3:
  return
}

// -----

"rd.pipeline_def"() ( {
  func @definition(%start: i64, %end: i64) -> !rd.Dataset {
    // CHECK: %[[RANGE:.*]] = rd.range %[[START]] to %[[END]]
    %0 = rd.range %start to %end by %start : (i64, i64, i64) -> !rd.Dataset
    // CHECK: %[[DOUBLED:.*]] = rd.inline_map @double(%[[RANGE]])
    %1 = rd.inline_map @double(%0) : (!rd.Dataset) -> !rd.Dataset
    // CHECK: %[[FILTERED:.*]] = rd.filter %[[DOUBLED]] excluding @less_than_five
    %2 = rd.filter %1 excluding @less_than_five : (!rd.Dataset) -> !rd.Dataset
    // CHECK: return %[[FILTERED]]
    return %2 : !rd.Dataset
  }
  rd.pipeline_def_terminator
}) {sym_name = "range_map_filter"}: () -> ()

func @use_dataset() {
  %start = constant 0 : i64
  %end = constant 5 : i64
  %itr = rd.make_iterator @range_map_filter(%start, %end) : (i64, i64) -> !rd.Iterator
  br ^bb1

^bb1:
  %isValid, %value = rd.iterator_next %itr : (!rd.Iterator) -> (i1, i64)
  cond_br %isValid, ^bb2, ^bb3

^bb2:
  rd.print %value : i64
  br ^bb1

^bb3:
  return
}