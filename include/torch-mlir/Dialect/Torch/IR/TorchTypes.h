//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_DIALECT_TORCH_IR_TORCHTYPES_H
#define TORCHMLIR_DIALECT_TORCH_IR_TORCHTYPES_H

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace torch {
namespace Torch {

/// PyTorch has a well-developed notion of subtyping.
///
/// This is a restricted subset of it that only handles a few special cases
/// that we need to model.
///
/// TODO: Flesh this out.
/// TODO: Decide / properly model the distinction between PEP 483 / Python
/// subtyping vs "more static information".
bool isValidSubtype(Type subtype, Type type);

class NonValueTensorType;
class ValueTensorType;

/// Common getter function signature that covers all tensor types.
/// Used for sharing code between NonValueTensorType and ValueTensorType.
using GetTensorTypeFn = llvm::function_ref<Type(
    MLIRContext *, std::optional<ArrayRef<int64_t>>, Type, Attribute)>;

/// The representation of an unknown dimension size in an ArrayRef<int64_t>.
constexpr static int64_t kUnknownSize = -1;

class BaseTensorType : public Type {
public:
  using Type::Type;

  /// Get the raw optional list of sizes.
  ///
  /// It is expected that for many users, `hasSizes`/`getSizes` will be a more
  /// convenient API.
  std::optional<ArrayRef<int64_t>> getOptionalSizes() const;

  /// Get the raw nullable Type representing the dtype of this tensor type.
  ///
  /// It is expected that for many users, `hasDtype`/`getDtype` will be a more
  /// convenient API.
  Type getOptionalDtype() const;

  /// Return true if this type has a list of sizes.
  bool hasSizes() const { return getOptionalSizes().has_value(); }

  /// Get the list of sizes. Requires `hasSizes()`.
  ArrayRef<int64_t> getSizes() const {
    assert(hasSizes() && "must have sizes");
    return getOptionalSizes().value();
  }

  /// Return true if all sizes of this tensor are known.
  bool areAllSizesKnown() const {
    return hasSizes() && llvm::all_of(getSizes(), [](int64_t size) {
             return size != kUnknownSize;
           });
  }

  /// Return true if this type has a known dtype.
  bool hasDtype() const { return static_cast<bool>(getOptionalDtype()); }

  /// Get the dtype. Requires `hasDtype()`.
  Type getDtype() const {
    assert(hasDtype() && "must have a dtype");
    return getOptionalDtype();
  }

  /// Enable isa/dyn_cast for BaseTensorType.
  static bool classof(Type type);

  /// Return true if this type has the same sizes and dtype as the other.
  bool hasSameSizesAndDtype(BaseTensorType other) const;

  /// Return a type of the same kind as this one, but with sizes and dtype
  /// from `other`.
  Type getWithSizesAndDtypeFrom(BaseTensorType other) const;

  /// Return a type of the same kind as this one, but with given raw optional
  /// sizes and raw optional dtype.
  Type getWithSizesAndDtype(std::optional<ArrayRef<int64_t>> optionalSizes,
                            Type optionalDtype) const;

  /// Return a type with the same shape and dtype as this one, but with
  /// value semantics.
  ValueTensorType getWithValueSemantics() const;
};

/// Return the tensor type which assumes the static information from both types.
///
/// For example, if `lhs = !torch.vtensor<[100],unk>` and
/// `rhs = !torch.vtensor<*,f32>` then this function would return
/// `!torch.vtensor<[100],f32>`.
///
/// Returns null if the types have conflicting static information.
///
/// This function requires both `lhs` and `rhs` to either both be
/// ValueTensorType or both be NonValueTensorType, since the sense of
/// "meet" between value tensors and non-value tensors is useful in different
/// ways in different situations.
Type meetTensorTypes(BaseTensorType lhs, BaseTensorType rhs);

} // namespace Torch
} // namespace torch
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h.inc"

//===----------------------------------------------------------------------===//
// Inline definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace torch {
namespace Torch {

inline std::optional<ArrayRef<int64_t>>
BaseTensorType::getOptionalSizes() const {
  if (auto tensor = dyn_cast<NonValueTensorType>())
    return tensor.getOptionalSizes();
  if (auto tensor = dyn_cast<ValueTensorType>())
    return tensor.getOptionalSizes();
  llvm_unreachable("not a BaseTensorType!");
}

inline Type BaseTensorType::getOptionalDtype() const {
  if (auto tensor = dyn_cast<NonValueTensorType>())
    return tensor.getOptionalDtype();
  if (auto tensor = dyn_cast<ValueTensorType>())
    return tensor.getOptionalDtype();
  llvm_unreachable("not a BaseTensorType!");
}

inline bool BaseTensorType::classof(Type type) {
  return type.isa<NonValueTensorType, ValueTensorType>();
}

} // namespace Torch
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCH_IR_TORCHTYPES_H
