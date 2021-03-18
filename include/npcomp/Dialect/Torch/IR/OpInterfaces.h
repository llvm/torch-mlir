//===- OpInterfaces.h -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_TORCH_IR_OPINTERFACES_H
#define NPCOMP_DIALECT_TORCH_IR_OPINTERFACES_H

#include "mlir/IR/Types.h"
#include "llvm/ADT/BitmaskEnum.h"

namespace mlir {
namespace NPCOMP {
namespace Torch {

/// Conversion rule to apply to a value (argument or return).
namespace KernelValueConversion {
enum BitMask : uint32_t {
  // No coercion.
  kNone = 0,

  // Coerce/require an immutable tensor value.
  kImmutableTensor = 2,

  // Coerce/require a mutable tensor value.
  kMutableTensor = 4,

  // If the source is a Scalar and the target is a Tensor, promotes
  // to a 0d tensor.
  kPromoteScalar = 8,

  // Drops the return value and aliases to argument 0.
  // TODO: Remove this in favor of general alias metadata processing (note that
  // the vast majority are this one case so it isn't so bad to have a special
  // case for it if necessary).
  kDropReturnAndAliasArg0 = 16,

  // Drops the argument/return.
  kDrop = 32,

  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ kDrop)
};
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();
} // namespace KernelValueConversion

/// Metadata for building a recognized/named kernel from a generic
/// torch.kernel_call.
struct KernelMetadata {
  /// The Torch kernel name.
  StringRef kernelName;

  /// Whether the kernel supports variable args or returns (extending the
  /// formal args and returns).
  bool isVararg = false;
  bool isVarret = false;

  /// Formal argument and return types in Torch signature syntax.
  SmallVector<StringRef, 4> argTypes;
  SmallVector<StringRef, 4> returnTypes;

  void addArgTypes(std::initializer_list<StringRef> ilist) {
    argTypes.insert(argTypes.end(), ilist);
  }

  void addReturnTypes(std::initializer_list<StringRef> ilist) {
    returnTypes.insert(returnTypes.end(), ilist);
  }
};

/// Extended metadata for constructing a kernel op.
struct BuildKernelMetadata : public KernelMetadata {
  /// Many ops have a variant with a single trailing parameter one argument
  /// past the formal argument list of the buildable op. If this flag is true,
  /// it activates a heuristic that will recognize this situation and promote
  /// it to an appropriate store of the result. It is deemed of little value
  /// to support each one of these variants as a dedicated op when they can
  /// all be handled with this flag.
  bool promoteTrailingOutTensor = false;

  /// Many ops have variant that treats the first (self) argument as an out
  /// param (usually denoted with a trailing `_`, such as `aten::div_`).
  /// When this string is set, it indicates the name of such a variant op.
  Optional<StringRef> inplaceVariantKernelName = None;

  SmallVector<KernelValueConversion::BitMask, 4> argConversions;
  SmallVector<KernelValueConversion::BitMask, 4> returnConversions;

  /// Additional alias kernel names to match.
  SmallVector<StringRef, 1> aliasKernelNames;

  void addArgConversions(
      std::initializer_list<KernelValueConversion::BitMask> ilist) {
    argConversions.insert(argConversions.end(), ilist);
  }
  void addReturnConversions(
      std::initializer_list<KernelValueConversion::BitMask> ilist) {
    returnConversions.insert(returnConversions.end(), ilist);
  }

  /// Gets the arg conversion flag for arg 'i'. Returns kNone if conversion
  /// flags are not defined that far.
  KernelValueConversion::BitMask getArgConversion(size_t i) const {
    if (i >= argConversions.size())
      return KernelValueConversion::BitMask::kNone;
    return argConversions[i];
  }
  /// Gets the return conversion flag for arg 'i'. Returns kNone if conversion
  /// flags are not defined that far.
  KernelValueConversion::BitMask getReturnConversion(size_t i) const {
    if (i >= returnConversions.size())
      return KernelValueConversion::BitMask::kNone;
    return returnConversions[i];
  }
};

} // namespace Torch

#include "npcomp/Dialect/Torch/IR/OpInterfaces.h.inc"
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_TORCH_IR_OPINTERFACES_H
