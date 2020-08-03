//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "npcomp/E2E/E2E.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"

#include "npcomp/Dialect/Npcomprt/IR/NpcomprtDialect.h"
#include "npcomp/Dialect/Npcomprt/IR/NpcomprtOps.h"
#include "npcomp/Dialect/TCP/IR/TCPOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

//===----------------------------------------------------------------------===//
// Creating module metadata.
//===----------------------------------------------------------------------===//

// Returns true if the function signature can be expressed with the npcomprt
// ABI.
static bool expressibleWithNpcomprtABI(FunctionType type) {
  // Currently, only tensor types can be exposed at npcomprt ABI boundaries.
  return llvm::all_of(
      llvm::concat<const Type>(type.getInputs(), type.getResults()),
      [](Type t) { return t.isa<TensorType>(); });
}

static LogicalResult createModuleMetadata(ModuleOp module) {
  auto moduleMetadata =
      OpBuilder::atBlockBegin(module.getBody())
          .create<npcomprt::ModuleMetadataOp>(module.getLoc());
  moduleMetadata.metadatas().push_back(new Block);
  Block &metadatas = moduleMetadata.metadatas().front();
  OpBuilder::atBlockEnd(&metadatas)
      .create<npcomprt::ModuleMetadataTerminatorOp>(module.getLoc());

  SymbolTable symbolTable(module);
  auto builder = OpBuilder::atBlockBegin(&metadatas);
  for (auto func : module.getOps<FuncOp>()) {
    if (symbolTable.getSymbolVisibility(func) !=
        SymbolTable::Visibility::Public) {
      continue;
    }
    // TODO: Add richer information here such as expected shapes and element
    // types.
    builder.create<npcomprt::FuncMetadataOp>(
        func.getLoc(), builder.getSymbolRefAttr(func.getName()),
        builder.getI32IntegerAttr(func.getNumArguments()),
        builder.getI32IntegerAttr(func.getNumResults()));

    if (!expressibleWithNpcomprtABI(func.getType()))
      return func.emitError() << "func not expressible with npcomprt ABI";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Dialect conversion.
//===----------------------------------------------------------------------===//

namespace {
class LowerTensorStoreOp : public OpConversionPattern<TensorStoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TensorStoreOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    TensorStoreOp::Adaptor adaptor(operands);
    auto memrefType = op.memref().getType().cast<MemRefType>();
    Value abiMemref = rewriter.create<npcomprt::ToMemrefOp>(
        op.getLoc(),
        UnrankedMemRefType::get(memrefType.getElementType(), /*memorySpace=*/0),
        adaptor.tensor());
    auto memref =
        rewriter.create<MemRefCastOp>(op.getLoc(), abiMemref, memrefType);
    rewriter.replaceOpWithNewOp<linalg::CopyOp>(op, memref, adaptor.memref());
    return success();
  }
};
} // namespace

namespace {
class LowerTensorLoadOp : public OpConversionPattern<TensorLoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TensorLoadOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    TensorLoadOp::Adaptor adaptor(operands);
    auto abiMemref = rewriter.create<MemRefCastOp>(
        op.getLoc(), adaptor.memref(),
        UnrankedMemRefType::get(
            adaptor.memref().getType().cast<MemRefType>().getElementType(),
            /*memorySpace=*/0));
    rewriter.replaceOpWithNewOp<npcomprt::FromMemrefOp>(
        op, rewriter.getType<npcomprt::TensorType>(), abiMemref);
    return success();
  }
};
} // namespace

namespace {
class LowerShapeOfOp : public OpConversionPattern<shape::ShapeOfOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(shape::ShapeOfOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    shape::ShapeOfOp::Adaptor adaptor(operands);
    // TODO: For now npcomp only supports ranked tensor types for its shape
    // lowering, since we don't have a runtime shape struct and lower all shapes
    // to individual SSA values.
    auto tensorType = op.arg().getType().cast<RankedTensorType>();
    SmallVector<Value, 6> extents;
    for (int i = 0, e = tensorType.getRank(); i < e; i++) {
      auto ci = rewriter.create<ConstantOp>(op.getLoc(),
                                            rewriter.getI32IntegerAttr(i));
      // TODO: Shouldn't the index type for the output be inferred since
      // https://reviews.llvm.org/rG31f40f603d0c00b313397196124c5f39090badf0
      // ?
      extents.push_back(rewriter.create<npcomprt::GetExtentOp>(
          op.getLoc(), rewriter.getIndexType(), adaptor.arg(), ci));
    }
    auto newShape = rewriter.create<shape::FromExtentsOp>(
        op.getLoc(), rewriter.getType<shape::ShapeType>(), extents);
    // TODO: Provide a builder that doesn't require the result type.
    rewriter.replaceOpWithNewOp<shape::ToExtentTensorOp>(
        op,
        RankedTensorType::get({ShapedType::kDynamicSize},
                              rewriter.getIndexType()),
        newShape);
    return success();
  }
};
} // namespace

namespace {
class LowerGlobalOp : public OpConversionPattern<tcp::GlobalOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tcp::GlobalOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<npcomprt::GlobalOp>(op, op.sym_name(),
                                                    op.value());
    return success();
  }
};
} // namespace

namespace {
class LowerGetGlobalMemrefOp
    : public OpConversionPattern<tcp::GetGlobalMemrefOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tcp::GetGlobalMemrefOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto abiMemrefType = UnrankedMemRefType::get(
        op.getType().cast<ShapedType>().getElementType(), /*memorySpace=*/0);
    auto abiMemref = rewriter.create<npcomprt::GetGlobalOp>(
        op.getLoc(), abiMemrefType, op.global());
    // Cast back to the original type.
    rewriter.replaceOpWithNewOp<MemRefCastOp>(op, abiMemref, op.getType());
    return success();
  }
};
} // namespace

static LogicalResult doDialectConversion(ModuleOp module) {
  auto *context = module.getContext();

  TypeConverter converter;
  converter.addConversion([](TensorType type) {
    return npcomprt::TensorType::get(type.getContext());
  });
  converter.addConversion([](npcomprt::TensorType type) { return type; });

  OwningRewritePatternList patterns;
  ConversionTarget target(*context);

  populateFuncOpTypeConversionPattern(patterns, context, converter);
  target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
    return converter.isSignatureLegal(op.getType());
  });

  patterns.insert<LowerTensorStoreOp>(context);
  target.addIllegalOp<TensorStoreOp>();
  target.addLegalOp<npcomprt::ToMemrefOp>();
  target.addLegalOp<linalg::CopyOp>();
  target.addLegalOp<MemRefCastOp>();

  patterns.insert<LowerTensorLoadOp>(context);
  target.addIllegalOp<TensorLoadOp>();
  target.addLegalOp<npcomprt::FromMemrefOp>();

  patterns.insert<LowerShapeOfOp>(context);
  target.addIllegalOp<shape::ShapeOfOp>();
  target.addLegalOp<ConstantOp>();
  target.addLegalOp<shape::FromExtentsOp>();
  target.addLegalOp<shape::ToExtentTensorOp>();
  target.addLegalOp<npcomprt::GetExtentOp>();

  patterns.insert<LowerGlobalOp>(context);
  target.addIllegalOp<tcp::GlobalOp>();
  target.addLegalOp<npcomprt::GlobalOp>();

  patterns.insert<LowerGetGlobalMemrefOp>(context);
  target.addIllegalOp<tcp::GetGlobalMemrefOp>();
  target.addLegalOp<npcomprt::GetGlobalOp>();

  return applyPartialConversion(module, target, patterns);
}

namespace {
// This pass lowers the public ABI of the module to the primitives exposed by
// the npcomprt dialect.
class LowerToNpcomprtABI : public LowerToNpcomprtABIBase<LowerToNpcomprtABI> {
  void runOnOperation() {
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
mlir::NPCOMP::createLowerToNpcomprtABIPass() {
  return std::make_unique<LowerToNpcomprtABI>();
}
