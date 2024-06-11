//===----------------------------------------------------------------------===//
//
// This source code is copied from PyTorch, and remains licensed under
// the PyTorch BSD-style license available at
// https://github.com/pytorch/pytorch/blob/master/LICENSE
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"

#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace torch {
namespace torch_upstream {

//===----------------------------------------------------------------------===//
// ScalarType enum related code are copied from c10/core/ScalarType.h.
//===----------------------------------------------------------------------===//
static inline bool isQIntType(ScalarType t) {
  // Don't forget to extend this when adding new QInt types
  return t == ScalarType::QInt8 || t == ScalarType::QUInt8 ||
         t == ScalarType::QInt32 || t == ScalarType::QUInt4x2 ||
         t == ScalarType::QUInt2x4 || t == ScalarType::QInt16;
}

//===----------------------------------------------------------------------===//
// Type promotion related code are copied from
// aten/src/ATen/native/TypeProperties.*.
//===----------------------------------------------------------------------===//
static inline ScalarType promoteTypes(ScalarType a, ScalarType b) {
  // This is generated according to NumPy's promote_types
  constexpr auto u1 = ScalarType::Byte;
  constexpr auto i1 = ScalarType::Char;
  constexpr auto i2 = ScalarType::Short;
  constexpr auto i4 = ScalarType::Int;
  constexpr auto i8 = ScalarType::Long;
  constexpr auto f2 = ScalarType::Half;
  constexpr auto f4 = ScalarType::Float;
  constexpr auto f8 = ScalarType::Double;
  constexpr auto c2 = ScalarType::ComplexHalf;
  constexpr auto c4 = ScalarType::ComplexFloat;
  constexpr auto c8 = ScalarType::ComplexDouble;
  constexpr auto b1 = ScalarType::Bool;
  constexpr auto bf = ScalarType::BFloat16;
  constexpr auto ud = ScalarType::Undefined;
  if (a == ud || b == ud) {
    return ScalarType::Undefined;
  }

  // For QInt types, we only allow exact match
  if (isQIntType(a) && a == b) {
    return a;
  }

  if (isQIntType(a) || isQIntType(b)) {
    assert(false && "promoteTypes with quantized numbers is not handled yet; "
                    "figure out what the correct rules should be");
  }

  // this matrix has to be consistent with AT_FORALL_SCALAR_TYPES_WITH_COMPLEX
  // so that's why we have to add undefined as we are not sure what is the
  // corrent values for the type promotions in complex type cases.
  static constexpr ScalarType _promoteTypesLookup[static_cast<int>(
      ScalarType::NumOptions)][static_cast<int>(ScalarType::NumOptions)] = {
      /*        u1  i1  i2  i4  i8  f2  f4  f8  c2  c4  c8  b1  q1  q2  q3  bf*/
      /* u1 */ {u1, i2, i2, i4, i8, f2, f4, f8, ud, c4, c8, u1, ud, ud, ud, bf},
      /* i1 */ {i2, i1, i2, i4, i8, f2, f4, f8, ud, c4, c8, i1, ud, ud, ud, bf},
      /* i2 */ {i2, i2, i2, i4, i8, f2, f4, f8, ud, c4, c8, i2, ud, ud, ud, bf},
      /* i4 */ {i4, i4, i4, i4, i8, f2, f4, f8, ud, c4, c8, i4, ud, ud, ud, bf},
      /* i8 */ {i8, i8, i8, i8, i8, f2, f4, f8, ud, c4, c8, i8, ud, ud, ud, bf},
      /* f2 */ {f2, f2, f2, f2, f2, f2, f4, f8, ud, c4, c8, f2, ud, ud, ud, f4},
      /* f4 */ {f4, f4, f4, f4, f4, f4, f4, f8, ud, c4, c8, f4, ud, ud, ud, f4},
      /* f8 */ {f8, f8, f8, f8, f8, f8, f8, f8, ud, c8, c8, f8, ud, ud, ud, f8},
      /* c2 */ {ud, ud, ud, ud, ud, ud, ud, ud, c2, c4, c8, ud, ud, ud, ud, ud},
      /* c4 */ {c4, c4, c4, c4, c4, c4, c4, c8, c4, c4, c8, c4, ud, ud, ud, c4},
      /* c8 */ {c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, ud, ud, ud, c8},
      /* b1 */ {u1, i1, i2, i4, i8, f2, f4, f8, ud, c4, c8, b1, ud, ud, ud, bf},
      /* q1 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
      /* q2 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
      /* q3 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
      /* bf */ {bf, bf, bf, bf, bf, f4, f4, f8, ud, c4, c8, bf, ud, ud, ud, bf},
  };
  return _promoteTypesLookup[static_cast<int>(a)][static_cast<int>(b)];
}

static inline bool isFloatingType(ScalarType t) {
  return (t == ScalarType::Double || t == ScalarType::Float ||
          t == ScalarType::Half || t == ScalarType::BFloat16);
}

static inline bool isComplexType(ScalarType t) {
  return (t == ScalarType::ComplexHalf || t == ScalarType::ComplexFloat ||
          t == ScalarType::ComplexDouble);
}

static inline ScalarType combine_categories(ScalarType higher,
                                            ScalarType lower) {
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (isComplexType(higher)) {
    return higher;
  } else if (!isComplexType(lower) && isFloatingType(higher)) {
    return higher;
  }
  if (higher == ScalarType::Bool || isFloatingType(lower) ||
      isComplexType(lower)) {
    return promote_skip_undefined(higher, lower);
  }
  if (higher != ScalarType::Undefined) {
    return higher;
  }
  return lower;
}

ScalarType promote_skip_undefined(ScalarType a, ScalarType b) {
  if (a == ScalarType::Undefined) {
    return b;
  }
  if (b == ScalarType::Undefined) {
    return a;
  }
  return promoteTypes(a, b);
}

ScalarType result_type(const ResultTypeState &in_state) {
  return combine_categories(
      in_state.dimResult,
      combine_categories(in_state.zeroResult, in_state.wrappedResult));
}

ReductionType get_reduction_enum(const llvm::StringRef &reduce) {
  if (reduce == "max" || reduce == "amax") {
    return torch_upstream::ReductionType::MAX;
  } else if (reduce == "mean") {
    return torch_upstream::ReductionType::MEAN;
  } else if (reduce == "min" || reduce == "amin") {
    return torch_upstream::ReductionType::MIN;
  } else if (reduce == "sum") {
    return torch_upstream::ReductionType::SUM;
  } else if (reduce == "prod") {
    return torch_upstream::ReductionType::PROD;
  } else {
    llvm_unreachable(
        "'reduce' argument must be either sum, prod, mean, amax or amin");
  }
}

} // namespace torch_upstream
} // namespace torch
} // namespace mlir
