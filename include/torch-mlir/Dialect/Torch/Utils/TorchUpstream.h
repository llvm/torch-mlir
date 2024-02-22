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
#include "llvm/ADT/StringRef.h"

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

// Note: As a coding convention, we should never `using` the `torch_upstream`
// namespace. This is to ensure that at a glance from the code, it is clear
// that we are referencing upstream types.

namespace mlir {
namespace torch {
namespace torch_upstream {

//===----------------------------------------------------------------------===//
// TypeKind related enum related code are copied from
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/jit_type_base.h
//===----------------------------------------------------------------------===//
#define C10_FORALL_TYPES(_)                                                    \
  _(AnyType)                                                                   \
  _(EnumType)                                                                  \
  _(AnyEnumType)                                                               \
  _(TensorType)                                                                \
  _(StorageType)                                                               \
  _(TupleType)                                                                 \
  _(ListType)                                                                  \
  _(DictType)                                                                  \
  _(NumberType)                                                                \
  _(FloatType)                                                                 \
  _(ComplexType)                                                               \
  _(FutureType)                                                                \
  _(RRefType)                                                                  \
  _(IntType)                                                                   \
  _(NoneType)                                                                  \
  _(StringType)                                                                \
  _(GeneratorType)                                                             \
  _(QuantizerType)                                                             \
  _(BoolType)                                                                  \
  _(OptionalType)                                                              \
  _(VarType)                                                                   \
  _(DeviceObjType)                                                             \
  _(StreamObjType)                                                             \
  _(FunctionType)                                                              \
  _(ClassType)                                                                 \
  _(PyObjectType)                                                              \
  _(CapsuleType)                                                               \
  _(InterfaceType)                                                             \
  _(QSchemeType)                                                               \
  _(LayoutType)                                                                \
  _(ScalarTypeType)                                                            \
  _(AnyListType)                                                               \
  _(AnyTupleType)                                                              \
  _(AnyClassType)                                                              \
  _(SymIntType)                                                                \
  _(UnionType)                                                                 \
  _(DynamicType)

enum class TypeKind {
#define DEFINE_TYPE(T) T,
  C10_FORALL_TYPES(DEFINE_TYPE)
#undef DEFINE_TYPE
};

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

//===----------------------------------------------------------------------===//
// Possible values for `memory_format` argument in PyTorch ops that support it.
// Source:
// https://github.com/pytorch/pytorch/blob/master/c10/core/MemoryFormat.h
//===----------------------------------------------------------------------===//
enum MemoryFormat { Contiguous, Preserve, ChannelsLast, ChannelsLast3d };

//===----------------------------------------------------------------------===//
// Possible values for `layout` argument in PyTorch ops that support it.
// Source:
// https://github.com/pytorch/pytorch/blob/master/c10/core/Layout.h
//===----------------------------------------------------------------------===//
enum Layout { Strided, Sparse, SparseCsr, Mkldnn, NumOptions };

//===----------------------------------------------------------------------===//
// Possible value for `EmbeddingBag Mode` argument for Embedding bag ops.
// Source:
// https://github.com/llvm/torch-mlir/blob/main/include/torch-mlir/Dialect/Torch/Utils/TorchUpstream.h
//===-----------------------------------------------------------------------===//
enum EmbeddingBagMode { MODE_SUM, MODE_MEAN, MODE_MAX };

//===----------------------------------------------------------------------===//
// Possible value for `reduce` argument for Scatter reduce ops.
// Source:
// https://github.com/llvm/torch-mlir/blob/main/include/torch-mlir/Dialect/Torch/Utils/TorchUpstream.h
//===-----------------------------------------------------------------------===//
enum ReductionType { MAX, MEAN, MIN, SUM, PROD };

ReductionType get_reduction_enum(const llvm::StringRef &reduce);

} // namespace torch_upstream
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCH_UPSTREAM_H
