// RUN: torch-mlir-opt %s -canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @scalar_floating_types(
// CHECK-SAME: %[[F32:.*]]: !torch.vtensor<[],f32>, %[[F16:.*]]: !torch.vtensor<[],f16>, %[[BF16:.*]]: !torch.vtensor<[],bf16>, %[[F64:.*]]: !torch.vtensor<[],f64>)
// CHECK-NEXT: return %[[F32]], %[[F16]], %[[BF16]], %[[F64]]
func.func @scalar_floating_types(%f32: !torch.vtensor<[],f32>, %f16: !torch.vtensor<[],f16>, %bf16: !torch.vtensor<[],bf16>, %f64: !torch.vtensor<[],f64>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[],f16>, !torch.vtensor<[],bf16>, !torch.vtensor<[],f64>) {
  %zero = torch.constant.int 0
  %negative_one = torch.constant.int -1
  %bfloat16 = torch.constant.int 15
  %none = torch.constant.none
  %0 = torch.aten.cumsum %f32, %zero, %none : !torch.vtensor<[],f32>, !torch.int, !torch.none -> !torch.vtensor<[],f32>
  %1 = torch.aten.cumsum %f16, %negative_one, %none : !torch.vtensor<[],f16>, !torch.int, !torch.none -> !torch.vtensor<[],f16>
  %2 = torch.aten.cumsum %bf16, %zero, %bfloat16 : !torch.vtensor<[],bf16>, !torch.int, !torch.int -> !torch.vtensor<[],bf16>
  %3 = torch.aten.cumsum %f64, %negative_one, %none : !torch.vtensor<[],f64>, !torch.int, !torch.none -> !torch.vtensor<[],f64>
  return %0, %1, %2, %3 : !torch.vtensor<[],f32>, !torch.vtensor<[],f16>, !torch.vtensor<[],bf16>, !torch.vtensor<[],f64>
}

// -----

// CHECK-LABEL: func.func @integer_promotion(
// CHECK: %[[RESULT:.*]] = torch.aten.cumsum
// CHECK-NEXT: return %[[RESULT]]
func.func @integer_promotion(%input: !torch.vtensor<[],si32>) -> !torch.vtensor<[],si64> {
  %zero = torch.constant.int 0
  %none = torch.constant.none
  %0 = torch.aten.cumsum %input, %zero, %none : !torch.vtensor<[],si32>, !torch.int, !torch.none -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}

// -----

// CHECK-LABEL: func.func @changed_dtype(
// CHECK: %[[RESULT:.*]] = torch.aten.cumsum
// CHECK-NEXT: return %[[RESULT]]
func.func @changed_dtype(%input: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f64> {
  %zero = torch.constant.int 0
  %float64 = torch.constant.int 7
  %0 = torch.aten.cumsum %input, %zero, %float64 : !torch.vtensor<[],f32>, !torch.int, !torch.int -> !torch.vtensor<[],f64>
  return %0 : !torch.vtensor<[],f64>
}

// -----

// CHECK-LABEL: func.func @runtime_dtype(
// CHECK: %[[RESULT:.*]] = torch.aten.cumsum
// CHECK-NEXT: return %[[RESULT]]
func.func @runtime_dtype(%input: !torch.vtensor<[],f32>, %dtype: !torch.optional<int>) -> !torch.vtensor<[],f32> {
  %zero = torch.constant.int 0
  %0 = torch.aten.cumsum %input, %zero, %dtype : !torch.vtensor<[],f32>, !torch.int, !torch.optional<int> -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL: func.func @unknown_element_type(
// CHECK: %[[RESULT:.*]] = torch.aten.cumsum
// CHECK-NEXT: return %[[RESULT]]
func.func @unknown_element_type(%input: !torch.vtensor<[],unk>) -> !torch.vtensor<[],unk> {
  %zero = torch.constant.int 0
  %none = torch.constant.none
  %0 = torch.aten.cumsum %input, %zero, %none : !torch.vtensor<[],unk>, !torch.int, !torch.none -> !torch.vtensor<[],unk>
  return %0 : !torch.vtensor<[],unk>
}

// -----

// CHECK-LABEL: func.func @invalid_dimensions(
// CHECK: %[[POSITIVE:.*]] = torch.aten.cumsum
// CHECK-NEXT: %[[NEGATIVE:.*]] = torch.aten.cumsum
// CHECK-NEXT: return %[[POSITIVE]], %[[NEGATIVE]]
func.func @invalid_dimensions(%input: !torch.vtensor<[],f32>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) {
  %one = torch.constant.int 1
  %negative_two = torch.constant.int -2
  %none = torch.constant.none
  %0 = torch.aten.cumsum %input, %one, %none : !torch.vtensor<[],f32>, !torch.int, !torch.none -> !torch.vtensor<[],f32>
  %1 = torch.aten.cumsum %input, %negative_two, %none : !torch.vtensor<[],f32>, !torch.int, !torch.none -> !torch.vtensor<[],f32>
  return %0, %1 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL: func.func @runtime_dimension(
// CHECK: %[[RESULT:.*]] = torch.aten.cumsum
// CHECK-NEXT: return %[[RESULT]]
func.func @runtime_dimension(%input: !torch.vtensor<[],f32>, %dim: !torch.int) -> !torch.vtensor<[],f32> {
  %none = torch.constant.none
  %0 = torch.aten.cumsum %input, %dim, %none : !torch.vtensor<[],f32>, !torch.int, !torch.none -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}

// -----

// CHECK-LABEL: func.func @nonvalue_tensor(
// CHECK: %[[RESULT:.*]] = torch.aten.cumsum
// CHECK-NEXT: return %[[RESULT]]
func.func @nonvalue_tensor(%input: !torch.tensor<[],f32>) -> !torch.tensor<[],f32> {
  %zero = torch.constant.int 0
  %none = torch.constant.none
  %0 = torch.aten.cumsum %input, %zero, %none : !torch.tensor<[],f32>, !torch.int, !torch.none -> !torch.tensor<[],f32>
  return %0 : !torch.tensor<[],f32>
}

// -----

// CHECK-LABEL: func.func @nonscalar_tensor(
// CHECK: %[[RESULT:.*]] = torch.aten.cumsum
// CHECK-NEXT: return %[[RESULT]]
func.func @nonscalar_tensor(%input: !torch.vtensor<[1],f32>) -> !torch.vtensor<[1],f32> {
  %zero = torch.constant.int 0
  %none = torch.constant.none
  %0 = torch.aten.cumsum %input, %zero, %none : !torch.vtensor<[1],f32>, !torch.int, !torch.none -> !torch.vtensor<[1],f32>
  return %0 : !torch.vtensor<[1],f32>
}

// -----

// CHECK-LABEL: func.func @unknown_rank(
// CHECK: %[[RESULT:.*]] = torch.aten.cumsum
// CHECK-NEXT: return %[[RESULT]]
func.func @unknown_rank(%input: !torch.vtensor<*,f32>) -> !torch.vtensor<*,f32> {
  %zero = torch.constant.int 0
  %none = torch.constant.none
  %0 = torch.aten.cumsum %input, %zero, %none : !torch.vtensor<*,f32>, !torch.int, !torch.none -> !torch.vtensor<*,f32>
  return %0 : !torch.vtensor<*,f32>
}
