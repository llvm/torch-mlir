//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "npcomp/RefBackend/RefBackend.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"

#include "npcomp/Dialect/Refback/IR/RefbackOps.h"
#include "npcomp/Dialect/Refbackrt/IR/RefbackrtDialect.h"
#include "npcomp/Dialect/Refbackrt/IR/RefbackrtOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

// Since input/output shapes are not hyper-rectangular we specify
// a maximum rank for each input shape such that shapes are padded
// out to kMaxRank at the ABI boundary. That way we can represent
// shapes using a traditional DenseElementsAttr.
// 
// NOTE: When changing this parameter, also change the same `kMaxRank`
// parameter in `lib/RefBackend/LowerToLLVM.cpp` so that the LLVM lowering
// stays consistent.
static constexpr int kMaxRank = 6;

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
      [](Type t) {
        return t.isa<UnrankedMemRefType, MemRefType, FloatType>();
      });
}

// Returns the integer rerpresentation of the CompilerDataStructures::ABIType
// Must stay aligned with CompilerDataStructures::ABIArgType enum
static uint32_t getIntReprForABIType(Type type) {
  if (type.isa<MemRefType>() || type.isa<UnrankedMemRefType>()) {
    return 1;
  } else if (auto floatTy = type.dyn_cast<FloatType>()) {
    switch (floatTy.getWidth()) {
    case 32:
      return 2;
    case 64:
      return 3;
    default:
      assert(false && "Unsupported float bit width");
    }
  } else if (auto intTy = type.dyn_cast<IntegerType>()) {
  }
  // assert(false && "couldn't get IntReprForABIType");
  return -1;
}

// Must stay aligned with CompilerDataStructures::ABIElementType enum
static uint32_t getIntReprForABIElementType(Type type) {
  if (auto shapedTy = type.dyn_cast<ShapedType>()) {
    auto elemTy = shapedTy.getElementType();
    if (auto floatTy = elemTy.dyn_cast<FloatType>()) {
      switch (floatTy.getWidth()) {
      case 32:
        return 1;
      default:
        assert(false && "Unsupported tensor element type");
      }
    }
  }
  return 0;
}

static SmallVector<int32_t, kMaxRank>
getExtentsForType(Type type, const int32_t maxRank = kMaxRank) {
  // Extend all shapes out to 4D to make our lives easier at the ABI boundary
  if (auto shapedType = type.dyn_cast<ShapedType>()) {
    if (!shapedType.hasRank()) {
      return {kMaxRank, kMaxRank, kMaxRank, kMaxRank, kMaxRank, kMaxRank};
    }

    auto shape = shapedType.getShape();
    auto shapeRank = shapedType.getRank();
    if (shapeRank <= maxRank) {
      SmallVector<int32_t, kMaxRank> extendedShape;
      // Push back all the values of the shape
      for (auto extentAndIndex : llvm::enumerate(shape)) {
        auto extent = extentAndIndex.value();
        auto index = extentAndIndex.index();
        if (shapedType.isDynamic(index)) {
          extendedShape.push_back(-1);
        } else {
          extendedShape.push_back(extent);
        }
      }

      // Pad whatever is left so we have even vectors
      auto padRank = maxRank - shapeRank;
      for (int i = 0; i < padRank; i++)
        extendedShape.push_back(0xDEAD'BEEF);

      return extendedShape;
    } else {
      assert(false && "unsupported rank");
    }
  }

  // Represent Scalar's as all 1's.
  return {kMaxRank, kMaxRank, kMaxRank, kMaxRank, kMaxRank, kMaxRank};
}

int32_t getRankForType(Type type) {
  // Returns a rank of -1 if the tensor is unranked
  if (auto shapedType = type.dyn_cast<ShapedType>()) {
    return shapedType.hasRank() ? shapedType.getRank() : -1;
  }
  return 0;
}

uint32_t hasStaticShape(Type type) {
  if (auto shapedType = type.dyn_cast<ShapedType>()) {
    return shapedType.hasStaticShape() ? 1 : 0;
  }
  // Assume scalars and non-shaped type things are static
  return 1;
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
    SmallVector<uint32_t, 6> inputABIArgTypes;
    SmallVector<uint32_t, 6> inputABIElementTypes;
    SmallVector<SmallVector<int32_t, kMaxRank>, 6> inputABIShapes;
    SmallVector<uint32_t, 6> inputABIRanks;
    // SmallVector<uint32_t, 6> inputIsStatic;
    for (const auto &inputArgType : func.getBody().front().getArgumentTypes()) {
      inputABIArgTypes.push_back(getIntReprForABIType(inputArgType));
      inputABIElementTypes.push_back(getIntReprForABIElementType(inputArgType));
      inputABIShapes.push_back(
          getExtentsForType(inputArgType, /*maxRank=*/kMaxRank));
      inputABIRanks.push_back(getRankForType(inputArgType));
      // inputIsStatic.push_back(hasStaticShape(inputArgType));
    }

    SmallVector<uint32_t, 6> outputABIArgTypes;
    SmallVector<uint32_t, 6> outputABIElementTypes;
    SmallVector<SmallVector<int32_t, kMaxRank>, 6> outputABIShapes;
    SmallVector<uint32_t, 6> outputABIRanks;
    SmallVector<uint32_t, 6> outputIsStatic;
    for (const auto &outputArgType : func.getCallableResults()) {
      outputABIArgTypes.push_back(getIntReprForABIType(outputArgType));
      outputABIElementTypes.push_back(
          getIntReprForABIElementType(outputArgType));
      outputABIShapes.push_back(
          getExtentsForType(outputArgType, /*maxRank=*/kMaxRank));
      outputABIRanks.push_back(getRankForType(outputArgType));
      // outputIsStatic.push_back(hasStaticShape(outputArgType));
    }

    auto i32Type = builder.getIntegerType(32);
    auto inputABIDataType =
        RankedTensorType::get(inputABIArgTypes.size(), i32Type);
    auto inputABIElementType =
        RankedTensorType::get(inputABIElementTypes.size(), i32Type);
    auto inputABIShapesType = RankedTensorType::get(
        llvm::ArrayRef<int64_t>{static_cast<long>(inputABIShapes.size()) *
                                kMaxRank},
        i32Type);
    auto inputABIRanksType =
        RankedTensorType::get(inputABIRanks.size(), i32Type);
    // auto inputIsStaticType = RankedTensorType::get(inputIsStatic.size(),
    // i32Type);
    auto outputABIDataType =
        RankedTensorType::get(outputABIArgTypes.size(), i32Type);
    auto outputABIElementType =
        RankedTensorType::get(outputABIElementTypes.size(), i32Type);
    auto outputABIShapesType = RankedTensorType::get(
        llvm::ArrayRef<int64_t>{static_cast<long>(outputABIShapes.size()) *
                                kMaxRank},
        i32Type);
    auto outputABIRanksType =
        RankedTensorType::get(outputABIRanks.size(), i32Type);
    // auto outputIsStaticType = RankedTensorType::get(outputIsStatic.size(),
    // i32Type);

    // TODO(brycearden): I'm sure there's a cleaner way to do this
    auto flattenABIShapes =
        [](SmallVector<SmallVector<int32_t, kMaxRank>, 6> shapes) {
          SmallVector<int32_t, 32> ret;
          for (auto &shape : shapes)
            for (auto &dim : shape)
              ret.push_back(dim);
          return ret;
        };

    SmallVector<NamedAttribute, 16> namedAttrs;

    // Add attributes that are valid for every func (funcName, numInputs,
    // numOutputs)
    namedAttrs.push_back(
        std::make_pair(Identifier::get("funcName", module.getContext()),
                       builder.getSymbolRefAttr(func.getName())));
    namedAttrs.push_back(
        std::make_pair(Identifier::get("numInputs", module.getContext()),
                       builder.getI32IntegerAttr(func.getNumArguments())));
    namedAttrs.push_back(
        std::make_pair(Identifier::get("numOutputs", module.getContext()),
                       builder.getI32IntegerAttr(func.getNumResults())));

    if (inputABIArgTypes.size()) {
      // Only add input information if there are inputs
      namedAttrs.push_back(std::make_pair(
          Identifier::get("inputArgTypes", func.getContext()),
          DenseIntElementsAttr::get(inputABIDataType,
                                    llvm::makeArrayRef(inputABIArgTypes))));
      namedAttrs.push_back(std::make_pair(
          Identifier::get("inputElementTypes", func.getContext()),
          DenseIntElementsAttr::get(inputABIElementType,
                                    llvm::makeArrayRef(inputABIElementTypes))));
      namedAttrs.push_back(std::make_pair(
          Identifier::get("inputRanks", func.getContext()),
          DenseIntElementsAttr::get(inputABIRanksType,
                                    llvm::makeArrayRef(inputABIRanks))));
      auto inputShapesFlattened = flattenABIShapes(inputABIShapes);
      namedAttrs.push_back(std::make_pair(
          Identifier::get("inputShapes", func.getContext()),
          DenseIntElementsAttr::get(
              inputABIShapesType,
              llvm::makeArrayRef(flattenABIShapes(inputABIShapes)))));
    }

    if (outputABIArgTypes.size()) {
      // Only add output information if there are outptus
      namedAttrs.push_back(std::make_pair(
          Identifier::get("outputArgTypes", func.getContext()),
          DenseIntElementsAttr::get(outputABIDataType,
                                    llvm::makeArrayRef(outputABIArgTypes))));
      namedAttrs.push_back(std::make_pair(
          Identifier::get("outputElementTypes", func.getContext()),
          DenseIntElementsAttr::get(
              outputABIElementType,
              llvm::makeArrayRef(outputABIElementTypes))));
      namedAttrs.push_back(std::make_pair(
          Identifier::get("outputRanks", func.getContext()),
          DenseIntElementsAttr::get(outputABIRanksType,
                                    llvm::makeArrayRef(outputABIRanks))));
      namedAttrs.push_back(std::make_pair(
          Identifier::get("outputShapes", func.getContext()),
          DenseIntElementsAttr::get(
              outputABIShapesType,
              llvm::makeArrayRef(flattenABIShapes(outputABIShapes)))));
    }

    builder.create<refbackrt::FuncMetadataOp>(func.getLoc(), ArrayRef<Type>{},
                                              ArrayRef<Value>{}, namedAttrs);

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
        auto memref = rewriter.create<memref::CastOp>(op.getLoc(), newArg,
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
        return builder.create<memref::CastOp>(
            loc, inputs[0], getABIMemrefType(inputs[0].getType()));
      });

  RewritePatternSet patterns(context);
  ConversionTarget target(*context);
  target.addLegalDialect<refbackrt::RefbackrtDialect>();
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<memref::MemRefDialect>();

  patterns.add<FuncOpSignatureConversion>(typeConverter, context);
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return typeConverter.isSignatureLegal(op.getType()); });
  patterns.add<RewriteReturnOp>(typeConverter, context);
  target.addDynamicallyLegalOp<ReturnOp>(
      [&](ReturnOp op) { return typeConverter.isLegal(op); });

  patterns.add<LowerAssertOp>(context);
  target.addIllegalOp<AssertOp>();

  return applyPartialConversion(module, target, std::move(patterns));
}

namespace {
// This pass lowers the public ABI of the module to the primitives exposed by
// the refbackrt dialect.
class LowerToRefbackrtABI
    : public LowerToRefbackrtABIBase<LowerToRefbackrtABI> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<refbackrt::RefbackrtDialect, memref::MemRefDialect>();
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
