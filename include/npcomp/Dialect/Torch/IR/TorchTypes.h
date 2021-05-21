//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_TORCH_IR_TORCHTYPES_H
#define NPCOMP_DIALECT_TORCH_IR_TORCHTYPES_H

#include "mlir/IR/Types.h"

namespace mlir {
namespace NPCOMP {
namespace Torch {

class NonValueTensorType;
class ValueTensorType;

/// Common getter function signature that covers all tensor types.
/// Used for sharing code between NonValueTensorType and ValueTensorType.
using GetTensorTypeFn =
    llvm::function_ref<Type(MLIRContext *, Optional<ArrayRef<int64_t>>, Type)>;

/// The representation of an unknown dimension size in an ArrayRef<int64_t>.
constexpr static int64_t kUnknownSize = -1;

class BaseTensorType : public Type {
public:
  using Type::Type;

  /// Get the raw optional list of sizes.
  ///
  /// It is expected that for many users, `hasSizes`/`getSizes` will be a more
  /// convenient API.
  Optional<ArrayRef<int64_t>> getOptionalSizes() const;

  /// Get the raw nullable Type representing the dtype of this tensor type.
  ///
  /// It is expected that for many users, `hasDtype`/`getDtype` will be a more
  /// convenient API.
  Type getOptionalDtype() const;

  /// Return true if this type has a list of sizes.
  bool hasSizes() const { return getOptionalSizes().hasValue(); }

  /// Get the list of sizes. Requires `hasSizes()`.
  ArrayRef<int64_t> getSizes() const {
    assert(hasSizes() && "must have sizes");
    return getOptionalSizes().getValue();
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
  Type getWithSizesAndDtype(Optional<ArrayRef<int64_t>> optionalSizes,
                            Type optionalDtype) const;

  /// Return a type with the same shape and dtype as this one, but with
  /// value semantics.
  ValueTensorType getWithValueSemantics() const;
};
} // namespace Torch
} // namespace NPCOMP
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "npcomp/Dialect/Torch/IR/TorchTypes.h.inc"

//===----------------------------------------------------------------------===//
// Inline definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace NPCOMP {
namespace Torch {

inline Optional<ArrayRef<int64_t>> BaseTensorType::getOptionalSizes() const {
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
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_TORCH_IR_TORCHTYPES_H
