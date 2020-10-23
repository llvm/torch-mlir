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
enum BitMask {
  // No coercion.
  kNone = 0,

  // Coerce/require an immutable tensor value.
  kImmutableTensor = 2,

  // Coerce/require a mutable tensor value.
  kMutableTensor = 4,

  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ kMutableTensor)
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

  SmallVector<KernelValueConversion::BitMask, 4> argConversions;
  SmallVector<KernelValueConversion::BitMask, 4> returnConversions;

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
