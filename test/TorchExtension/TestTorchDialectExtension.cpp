//===- TestTorchDialectExtension.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an extension of the Torch dialect for testing
// purposes.
//
//===----------------------------------------------------------------------===//

#include "TestTorchDialectExtension.h"
#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"
#include "mlir/Pass/Pass.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/UtilsForODSGenerated.h"

using namespace mlir;

namespace {

/// Test extension of the Torch dialect. Registers additional ops and
class TestTorchDialectExtension
    : public mlir::torch::Torch::TorchDialectExtension<
          TestTorchDialectExtension> {
public:
  using Base::Base;

  void init() {
    registerTorchOps<
#define GET_OP_LIST
#include "TestTorchDialectExtension.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "TestTorchDialectExtension.cpp.inc"

namespace {

/// This pass applies the permutation on the first maximal perfect nest.
struct TestTorchExtensionOpPass
    : public PassWrapper<TestTorchExtensionOpPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTorchExtensionOpPass)

  StringRef getArgument() const final { return "test-torch-dialect-extension"; }
  TestTorchExtensionOpPass() = default;
  TestTorchExtensionOpPass(const TestTorchExtensionOpPass &pass)
      : PassWrapper(pass){};

  void runOnOperation() override;
};

} // namespace

void TestTorchExtensionOpPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  getOperation()->walk([&](mlir::test::GoofyIdentityOp goofyOp) {
    goofyOp->setAttr("bob", StringAttr::get(ctx, StringRef("uncle")));
  });
}

void ::test::registerTestTorchDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<TestTorchDialectExtension>();
  PassRegistration<TestTorchExtensionOpPass>();
}
