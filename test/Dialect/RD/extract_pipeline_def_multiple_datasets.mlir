// RUN: npcomp-opt -rd-extract-pipeline-def %s | FileCheck --dump-input=fail %s

func @double(%input: i64) -> i64 {
  %0 = addi %input, %input : i64
  return %0 : i64
}

func @less_than_five(%input: i64) -> i1 {
  %five = constant 5 : i64
  %result = cmpi "slt", %input, %five : i64
  return %result : i1
}

func @range_map_filter(%start: i64, %end: i64) -> !rd.Dataset {
  %0 = rd.range %start to %end by %start : (i64, i64, i64) -> !rd.Dataset
  %1 = rd.inline_map @double(%0) : (!rd.Dataset) -> !rd.Dataset
  %2 = rd.filter %1 excluding @less_than_five : (!rd.Dataset) -> !rd.Dataset
  return %2 : !rd.Dataset
}
// CHECK-NOT: func @range_map_filter
// CHECK-LABEL: "rd.pipeline_def"
// CHECK: func @definition(%[[START:.*]]: i64, %[[END:.*]]: i64) -> !rd.Dataset {
// CHECK:   %[[RANGE:.*]] = rd.range %[[START]] to %[[END]] by %[[START]]
// CHECK:   %[[MAPPED:.*]] = rd.inline_map @double(%[[RANGE]])
// CHECK:   %[[FILTERED:.*]] = rd.filter %[[MAPPED]] excluding @less_than_five
// CHECK:   return %[[FILTERED]]
// CHECK: sym_name = "range_map_filter"
// CHECK-NOT: func @range_map_filter
// CHECK: func @use_dataset
// CHECK-NOT: func @range_map_filter

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

func @range_map_filter2(%start: i64, %end: i64) -> !rd.Dataset {
  %0 = rd.range %start to %end by %start : (i64, i64, i64) -> !rd.Dataset
  %1 = rd.inline_map @double(%0) : (!rd.Dataset) -> !rd.Dataset
  %2 = rd.filter %1 excluding @less_than_five : (!rd.Dataset) -> !rd.Dataset
  return %2 : !rd.Dataset
}
// CHECK-NOT: func @range_map_filter2
// CHECK-LABEL: "rd.pipeline_def"
// CHECK: func @definition(%[[START:.*]]: i64, %[[END:.*]]: i64) -> !rd.Dataset {
// CHECK:   %[[RANGE:.*]] = rd.range %[[START]] to %[[END]] by %[[START]]
// CHECK:   %[[MAPPED:.*]] = rd.inline_map @double(%[[RANGE]])
// CHECK:   %[[FILTERED:.*]] = rd.filter %[[MAPPED]] excluding @less_than_five
// CHECK:   return %[[FILTERED]]
// CHECK: sym_name = "range_map_filter2"
// CHECK-NOT: func @range_map_filter2
// CHECK: func @use_dataset
// CHECK-NOT: func @range_map_filter2

func @use_dataset2() {
  %start = constant 0 : i64
  %end = constant 5 : i64
  %itr = rd.make_iterator @range_map_filter2(%start, %end) : (i64, i64) -> !rd.Iterator
  return
}


// Ensure that a function that is unused in dataset ops is not transformed.
// CHECK-LABEL: func @unused_dataset
func @unused_dataset(%start: i64, %end: i64) -> !rd.Dataset {
  %0 = rd.range %start to %end by %start : (i64, i64, i64) -> !rd.Dataset
  return %0 : !rd.Dataset
}
