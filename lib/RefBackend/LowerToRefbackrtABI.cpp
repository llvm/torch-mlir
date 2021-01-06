//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "npcomp/RefBackend/RefBackend.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"

#include "npcomp/Dialect/Refback/IR/RefbackOps.h"
#include "npcomp/Dialect/Refbackrt/IR/RefbackrtDialect.h"
#include "npcomp/Dialect/Refbackrt/IR/RefbackrtOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

// Get the type used to represent MemRefType `type` on ABI boundaries.
// For convenience we do a cast to MemRefType internally.
static Type getABIMemrefType(Type type) {
  return UnrankedMemRefType::get(type.cast<MemRefType>().getElementType(),
                                 /*memorySpace=*/0);
}

//===----------------------------------------------------------------------===//
// Creating module metadata.
//===----------------------------------------------------------------------===//

// Returns true if the function signature can be expressed with the refbackrt
// ABI.
static bool expressibleWithRefbackrtABI(FunctionType type) {
  // Currently, only memref types can be exposed at refbackrt ABI boundaries.
  return llvm::all_of(
      llvm::concat<const Type>(type.getInputs(), type.getResults()),
      [](Type t) { return t.isa<MemRefType>(); });
}

static LogicalResult createModuleMetadata(ModuleOp module) {
  auto moduleMetadata =
      OpBuilder::atBlockBegin(module.getBody())
          .create<refbackrt::ModuleMetadataOp>(module.getLoc());
  moduleMetadata.metadatas().push_back(new Block);
  Block &metadatas = moduleMetadata.metadatas().front();
  OpBuilder::atBlockEnd(&metadatas)
      .create<refbackrt::ModuleMetadataTerminatorOp>(module.getLoc());

  SymbolTable symbolTable(module);
  auto builder = OpBuilder::atBlockBegin(&metadatas);
  for (auto func : module.getOps<FuncOp>()) {
    if (symbolTable.getSymbolVisibility(func) !=
        SymbolTable::Visibility::Public) {
      continue;
    }
    // TODO: Add richer information here such as expected shapes and element
    // types.
    builder.create<refbackrt::FuncMetadataOp>(
        func.getLoc(), builder.getSymbolRefAttr(func.getName()),
        builder.getI32IntegerAttr(func.getNumArguments()),
        builder.getI32IntegerAttr(func.getNumResults()));

    if (!expressibleWithRefbackrtABI(func.getType()))
      return func.emitError() << "func not expressible with refbackrt ABI";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Dialect conversion.
//===----------------------------------------------------------------------===//

namespace {
class LowerAssertOp : public OpConversionPattern<AssertOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AssertOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    AssertOp::Adaptor adaptor(operands);
    // The refbackrt runtime function aborts if the argument is true, rather
    // than when it is false as an `assert` does. So negate the predicate (by
    // xor'ing with 1).
    auto c1 = rewriter.create<ConstantOp>(
        op.getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(),
                                             APInt(/*numBits=*/1, /*val=*/1)));
    Value assertFailed = rewriter.create<XOrOp>(op.getLoc(), adaptor.arg(), c1);
    rewriter.replaceOpWithNewOp<refbackrt::AbortIfOp>(op, assertFailed,
                                                      op.msgAttr());
    return success();
  }
};
} // namespace

namespace {
// At ABI boundaries, convert all memrefs to unranked memrefs so that they have
// a fixed ABI.
class FuncOpSignatureConversion : public OpConversionPattern<FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FuncOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType type = op.getType();

    TypeConverter::SignatureConversion entryConversion(type.getNumInputs());
    if (failed(typeConverter->convertSignatureArgs(type.getInputs(),
                                                   entryConversion)))
      return rewriter.notifyMatchFailure(op, "could not convert inputs");
    SmallVector<Type, 1> newResultTypes;
    if (failed(typeConverter->convertTypes(type.getResults(), newResultTypes)))
      return rewriter.notifyMatchFailure(op, "could not convert outputs");

    rewriter.updateRootInPlace(op, [&] {
      // Update the function type.
      op.setType(FunctionType::get(op.getContext(),
                                   entryConversion.getConvertedTypes(),
                                   newResultTypes));
      // Rewrite the entry block.
      Block &oldEntry = op.getBody().front();
      Block &newEntry =
          *rewriter.applySignatureConversion(&op.getBody(), entryConversion);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&newEntry);
      BlockArgument newArg, oldArg;
      for (auto newAndOldArg :
           llvm::zip(newEntry.getArguments(), oldEntry.getArguments())) {
        std::tie(newArg, oldArg) = newAndOldArg;
        auto memref = rewriter.create<MemRefCastOp>(op.getLoc(), newArg,
                                                    oldArg.getType());
        rewriter.replaceUsesOfBlockArgument(oldArg, memref);
      }
    });
    return success();
  }
};
} // namespace

namespace {
// At the return ABI boundaries, convert to the ABI type.
// This pattern is needed to trigger the type conversion mechanics to do a
// target materialization.
class RewriteReturnOp : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReturnOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ReturnOp>(op, operands);
    return success();
  }
};
} // namespace

static LogicalResult doDialectConversion(ModuleOp module) {
  auto *context = module.getContext();

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion(
      [](MemRefType type) { return getABIMemrefType(type); });
  typeConverter.addTargetMaterialization(
      [](OpBuilder &builder, UnrankedMemRefType type, ValueRange inputs,
         Location loc) -> Value {
        assert(inputs.size() == 1);
        return builder.create<MemRefCastOp>(
            loc, inputs[0], getABIMemrefType(inputs[0].getType()));
      });

  OwningRewritePatternList patterns;
  ConversionTarget target(*context);
  target.addLegalDialect<refbackrt::RefbackrtDialect>();
  target.addLegalDialect<StandardOpsDialect>();

  patterns.insert<FuncOpSignatureConversion>(typeConverter, context);
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return typeConverter.isSignatureLegal(op.getType()); });
  patterns.insert<RewriteReturnOp>(typeConverter, context);
  target.addDynamicallyLegalOp<ReturnOp>(
      [&](ReturnOp op) { return typeConverter.isLegal(op); });

  patterns.insert<LowerAssertOp>(context);
  target.addIllegalOp<AssertOp>();

  return applyPartialConversion(module, target, std::move(patterns));
}

namespace {
// This pass lowers the public ABI of the module to the primitives exposed by
// the refbackrt dialect.
class LowerToRefbackrtABI
    : public LowerToRefbackrtABIBase<LowerToRefbackrtABI> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<refbackrt::RefbackrtDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Before we lower anything, capture any needed metadata about the argument
    // lists that will be needed for safely invoking the raw runtime functions
    // later. (for example, number of expected arguments/results, types,
    // etc.)
    if (failed(createModuleMetadata(module)))
      return signalPassFailure();

    // Now do the actual conversion / lowering.
    if (failed(doDialectConversion(module)))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::NPCOMP::createLowerToRefbackrtABIPass() {
  return std::make_unique<LowerToRefbackrtABI>();
}
