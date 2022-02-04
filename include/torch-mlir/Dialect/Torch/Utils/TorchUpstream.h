//===----------------------------------------------------------------------===//
//
// This source code is copied from PyTorch, and remains licensed under
// the PyTorch BSD-style license available at
// https://github.com/pytorch/pytorch/blob/master/LICENSE
//
//===----------------------------------------------------------------------===//
#ifndef TORCHMLIR_DIALECT_TORCH_UPSTREAM_H
#define TORCHMLIR_DIALECT_TORCH_UPSTREAM_H

#include "mlir/Support/LLVM.h"

// For layering reasons, the parts of the core MLIR compiler code written in C++
// never take a C++ dependency on Torch itself (any code depending on Torch C++
// API should be using the Torch-MLIR CAPI). However, certain highly stable enum
// values and logic are minimally needed to match PyTorch semantics, which we
// choose to copy into our codebase. The amount of code copied here should be
// kept to the absolute minimum and be restricted to highly stable logic and
// never leak out to the rest of the codebase. The code should be copied
// verbatim (modulo namespaces) from PyTorch. Notice that this file retains the
// original PyTorch license and the code here should not be mixed with "code
// that we [Torch-MLIR] write".

namespace mlir {
namespace torch {
namespace torch_upstream {

//===----------------------------------------------------------------------===//
// ScalarType enum related code are copied from c10/core/ScalarType.h
//===----------------------------------------------------------------------===//

// at:: and c10:: parts of the macro are never used within the compiler -- we
// only use this for the enum values.
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(_)                       \
  _(uint8_t, Byte)                        /* 0 */                              \
  _(int8_t, Char)                         /* 1 */                              \
  _(int16_t, Short)                       /* 2 */                              \
  _(int, Int)                             /* 3 */                              \
  _(int64_t, Long)                        /* 4 */                              \
  _(at::Half, Half)                       /* 5 */                              \
  _(float, Float)                         /* 6 */                              \
  _(double, Double)                       /* 7 */                              \
  _(c10::complex<c10::Half>, ComplexHalf) /* 8 */                              \
  _(c10::complex<float>, ComplexFloat)    /* 9 */                              \
  _(c10::complex<double>, ComplexDouble)  /* 10 */                             \
  _(bool, Bool)                           /* 11 */                             \
  _(c10::qint8, QInt8)                    /* 12 */                             \
  _(c10::quint8, QUInt8)                  /* 13 */                             \
  _(c10::qint32, QInt32)                  /* 14 */                             \
  _(at::BFloat16, BFloat16)               /* 15 */                             \
  _(c10::quint4x2, QUInt4x2)              /* 16 */                             \
  _(c10::quint2x4, QUInt2x4)              /* 17 */

enum class ScalarType : int8_t {
#define DEFINE_ENUM(_1, n) n,
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ENUM)
#undef DEFINE_ENUM
      Undefined,
  NumOptions
};

//===----------------------------------------------------------------------===//
// Type promotion related functions and struct definitions are copied from
// aten/src/ATen/native/TypeProperties.*
//===----------------------------------------------------------------------===//
struct ResultTypeState {
  ScalarType dimResult = ScalarType::Undefined;
  ScalarType wrappedResult = ScalarType::Undefined;
  ScalarType zeroResult = ScalarType::Undefined;
};

ScalarType result_type(const ResultTypeState &in_state);
ScalarType promote_skip_undefined(ScalarType a, ScalarType b);

//===----------------------------------------------------------------------===//
// These constants control the reduction behavior of the loss functions.
// None, Mean and Sum corresponds to "do not reduce", "Mean of losses", and "sum
// of losses" respectively.
// Source:
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/Reduction.h
//===----------------------------------------------------------------------===//
enum Reduction { None, Mean, Sum, END };

} // namespace torch_upstream
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCH_UPSTREAM_H
