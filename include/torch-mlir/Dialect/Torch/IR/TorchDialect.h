//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_DIALECT_TORCH_IR_TORCHDIALECT_H
#define TORCHMLIR_DIALECT_TORCH_IR_TORCHDIALECT_H

#include "mlir/IR/Dialect.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h.inc"

namespace mlir {
namespace torch {
namespace Torch {

/// Parse a type registered to this dialect.
Type parseTorchDialectType(AsmParser &parser);

/// Print a type registered to this dialect.
void printTorchDialectType(Type type, AsmPrinter &printer);

/// Base class for extensions of the Torch dialect that supports injecting
/// operations into the Torch dialect at load time. Concrete extensions are
/// expected to derive this class and register operations in the constructor.
/// They can be registered with the DialectRegistry and automatically applied
/// to the Torch dialect when it is loaded.
///
/// Derived classes are expected to define a `void init()` function in which
/// they can call various protected methods of the base class to register
/// extension operations and declare their dependencies.
template <typename DerivedTy, typename... ExtraDialects>
class TorchDialectExtension
    : public DialectExtension<DerivedTy, TorchDialect, ExtraDialects...> {
  using Initializer = std::function<void(TorchDialect *)>;

public:
  /// Extension application hook. Actually loads the dependent dialects and
  /// registers the additional operations. Not expected to be called directly.
  void apply(MLIRContext *context, TorchDialect *torchDialect,
             ExtraDialects *...) const final {
    for (const Initializer &init : opInitializers)
      init(torchDialect);
  }

protected:
  using Base = TorchDialectExtension<DerivedTy, ExtraDialects...>;

  /// Extension constructor. The argument indicates whether to skip generated
  /// dialects when applying the extension.
  explicit TorchDialectExtension() { static_cast<DerivedTy *>(this)->init(); }

  /// Hook for derived classes to inject constructor behavior.
  void init() {}

  /// Injects the operations into the Torch dialect.
  template <typename... OpTys> void registerTorchOps() {
    opInitializers.push_back([](TorchDialect *transformDialect) {
      transformDialect->addOperationsChecked<OpTys...>();
    });
  }

private:
  SmallVector<Initializer> opInitializers;
};

template <typename OpTy> void TorchDialect::addOperationIfNotRegistered() {
  StringRef name = OpTy::getOperationName();
  std::optional<RegisteredOperationName> opName =
      RegisteredOperationName::lookup(name, getContext());
  if (!opName) {
    addOperations<OpTy>();
    return;
  }

  if (opName->getTypeID() == TypeID::get<OpTy>())
    return;

  reportDuplicateOpRegistration(name);
}

} // namespace Torch
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCH_IR_TORCHDIALECT_H
