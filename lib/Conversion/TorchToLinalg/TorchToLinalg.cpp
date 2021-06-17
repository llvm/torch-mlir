//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Conversion/TorchToLinalg/TorchToLinalg.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h" // TODO: For `memref.dim`.
#include "mlir/Dialect/Traits.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/Torch/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

// -----------------------------------------------------------------------------
// Patterns (as this grows, it should be organized into multiple files)
// -----------------------------------------------------------------------------
// This is going to eventually be O(#aten ops), which is in the 100s.
//
// Most of these patterns consist of:
// 1. Checking that the operand/result types and other static properties are
//    good-enough to create a valid linalg op (such as operands being of
//    ranks/dtypes acceptable to the linalg op).
// 2. Creating dynamic error guards, usually checking a predicate on the
//    compatibility of operand shapes.
// 3. Creating init tensors for the computation op. Usually this involves
//    reifying IR for a shape transfer function based on the operand shapes.
// 4. Creating a named linalg op to replace the original op.
//
// TODO: Use linalg OpDSL to autogenerate at least 1)/2)/3) such
// that these patterns become mostly mechanical associations of
// "aten.foo -> linalg.foo".

static LogicalResult verifyLinalgCompatibleTypes(Operation *op,
                                                 PatternRewriter &rewriter) {
  // For now, use a small allowlist of types we don't reject.
  // The main culprit in practice is an unknown dtype
  // when RefineTypes isn't smart enough to propagate it everywhere.
  // For tensors, we consider the post-conversion tensor type (this pass is
  // doing a type conversion).
  auto isValidLinalgType = [](Type type) {
    if (auto tensor = type.dyn_cast<ValueTensorType>()) {
      if (auto rankedTensor =
              tensor.toBuiltinTensor().dyn_cast_or_null<RankedTensorType>()) {
        if (BaseMemRefType::isValidElementType(rankedTensor.getElementType()))
          return true;
      }
    }
    if (type.isa<mlir::FloatType, IntegerType, IndexType>())
      return true;
    return false;
  };
  bool valid = llvm::all_of(op->getOperandTypes(), isValidLinalgType) &&
               llvm::all_of(op->getResultTypes(), isValidLinalgType);
  if (!valid)
    return rewriter.notifyMatchFailure(op, "type cannot be lowered to linalg");
  return success();
}

namespace {
class ConvertAtenMmOp : public OpConversionPattern<AtenMmOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMmOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value lhs = operands[0];
    Value rhs = operands[1];

    // A user can write an errorneous program where `aten.mm` is in fact called
    // with operands of invalid rank or dtype. We cannot convert to linalg in
    // this case or we will get a verifier error, which corresponds to breaking
    // of *internal* compiler invariants, and for a user manifests as a compiler
    // crash in the worst case (such as we try to canonicalize/fold/print the
    // invalid op before the verifier gets to see it -- also release builds of a
    // mature copmiler usually have the verifier turned off for compile time
    // reasons).
    //
    // The compiler cannot crash even if the user wrote an erroneous program!
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    if (lhs.getType().cast<RankedTensorType>().getRank() != 2 ||
        rhs.getType().cast<RankedTensorType>().getRank() != 2) {
      return rewriter.notifyMatchFailure(
          op, "expected both operands to aten.mm to be rank 2");
    }

    Value lhsDim0 = rewriter.create<memref::DimOp>(loc, lhs, 0);
    Value lhsDim1 = rewriter.create<memref::DimOp>(loc, lhs, 1);
    Value rhsDim0 = rewriter.create<memref::DimOp>(loc, rhs, 0);
    Value rhsDim1 = rewriter.create<memref::DimOp>(loc, rhs, 1);
    Value contractingDimEqual =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, lhsDim1, rhsDim0);
    rewriter.create<AssertOp>(
        loc, contractingDimEqual,
        rewriter.getStringAttr(
            "mismatching contracting dimension for torch.aten.mm"));

    Type newResultType = getTypeConverter()->convertType(op.getType());
    Type elementType = newResultType.cast<TensorType>().getElementType();
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ValueRange{lhsDim0, rhsDim1}, elementType);
    Value c0 =
        rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0.0));
    Value zeroFill =
        rewriter.create<linalg::FillOp>(loc, initTensor, c0).getResult(0);
    Value matmul = rewriter
                       .create<linalg::MatmulOp>(loc, zeroFill.getType(),
                                                 ValueRange{lhs, rhs}, zeroFill)
                       .getResult(0);
    // When constructed with just dynamic sizes, InitTensorOp will have a result
    // type which has all `?`'s for dimensions, which might not be the result
    // type of `op`. The constraints on later linalg ops means that the result
    // of the MatmulOp will have this type too. So cast it to the desired type
    // so that in the end we have the original result type.
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, matmul);

    return success();
  }
};
} // namespace

namespace {
// See comments at in convertMmOp and the heading for this section for general
// considerations. This function needs to be auto-generated.
class ConvertAtenLinearOp : public OpConversionPattern<AtenLinearOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenLinearOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    AtenLinearOp::Adaptor adaptor(operands);
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();
    Value input = adaptor.input();
    Value weight = adaptor.weight();
    Value bias = adaptor.bias();
    // TODO: Handle the case of bias being None (bias is optional).
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    auto inputType = input.getType().cast<RankedTensorType>();
    auto weightType = weight.getType().cast<RankedTensorType>();
    auto biasType = bias.getType().cast<RankedTensorType>();
    // Only handle the case of rank 2 `input` for now.
    // TODO: Insert the appropriate reshape to collapse any leading dimensions.
    if (inputType.getRank() != 2 || weightType.getRank() != 2 ||
        biasType.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          op,
          "expected both input and weight to be rank 2 and bias to be rank 1");
    }
    // TODO: Handle type promotion. What are ATen's promotion rules?
    if (inputType.getElementType() != weightType.getElementType() ||
        inputType.getElementType() != biasType.getElementType()) {
      return rewriter.notifyMatchFailure(op, "unimplemented: type promotion");
    }

    // TODO: We can handle a static size 1 here at some complexity cost, but the
    // dynamic case is not representable in linalg. We don't handle either for
    // now. Biases are generally statically shaped for most models (since for
    // inference they are constants, and for training they don't change shape
    // typically), so this is not too constraining.
    auto biasSize = bias.getType().cast<RankedTensorType>().getShape()[0];
    if (biasSize == 1 || biasSize == ShapedType::kDynamicSize)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: size-1 broadcasting for aten::LinearOp");

    auto getDimOp = [&](Value v, int dimension) {
      return rewriter.create<memref::DimOp>(loc, v, dimension);
    };
    Value inputDim0 = getDimOp(input, 0);
    Value inputDim1 = getDimOp(input, 1);
    Value weightDim0 = getDimOp(weight, 0);
    Value weightDim1 = getDimOp(weight, 1);
    Value biasDim0 = getDimOp(bias, 0);
    Value contractingDimEqual =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, inputDim1, weightDim1);
    rewriter.create<AssertOp>(
        loc, contractingDimEqual,
        rewriter.getStringAttr(
            "mismatching contracting dimension for aten.linear"));
    // Here we take advantage of ruling out the size-1 case above.
    // In the static-size-1 case, we will not emit this check at all.
    Value biasSizeCorrect =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, weightDim0, biasDim0);
    rewriter.create<AssertOp>(
        loc, biasSizeCorrect,
        rewriter.getStringAttr("mismatching bias size for aten.linear"));

    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ValueRange{inputDim0, weightDim0}, inputType.getElementType());
    SmallVector<AffineMap> broadcastIndexingMaps = {
        AffineMap::get(
            /*dimCount=*/2, /*symbolCount=*/0, rewriter.getAffineDimExpr(1)),
        rewriter.getMultiDimIdentityMap(2)};
    SmallVector<StringRef> iteratorTypes(2, "parallel");
    Value broadcasted =
        rewriter
            .create<linalg::GenericOp>(
                loc, initTensor.getType(), bias, initTensor,
                /*indexingMaps=*/broadcastIndexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [](OpBuilder &b, Location loc, ValueRange args) {
                  b.create<linalg::YieldOp>(loc, args[0]);
                })
            .getResult(0);
    // We need a matmul with dimension ordering (N, K) * (M, K), so transpose
    // the weights to fit into linalg::MatmulOp which is (N, K) * (K, M).
    // TODO: This whole aten.linear lowering should eventually be generated from
    // a single linalg ODS generator statement. Both the bias and matmul part.
    SmallVector<AffineMap> transposeIndexingMaps = {
        AffineMap::get(
            /*dimCount=*/2, /*symbolCount=*/0,
            {rewriter.getAffineDimExpr(1), rewriter.getAffineDimExpr(0)},
            context),
        rewriter.getMultiDimIdentityMap(2)};
    Value transposedWeightInitTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ValueRange{weightDim1, weightDim0}, weightType.getElementType());
    Value transposedWeights =
        rewriter
            .create<linalg::GenericOp>(
                loc, transposedWeightInitTensor.getType(), weight,
                transposedWeightInitTensor,
                /*indexingMaps=*/transposeIndexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [](OpBuilder &b, Location loc, ValueRange args) {
                  b.create<linalg::YieldOp>(loc, args[0]);
                })
            .getResult(0);
    Value matmul = rewriter
                       .create<linalg::MatmulOp>(
                           loc, broadcasted.getType(),
                           ValueRange{input, transposedWeights}, broadcasted)
                       .getResult(0);
    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, matmul);
    return success();
  }
};
} // namespace

static Value createScalarRelu(OpBuilder &b, Location loc, ValueRange args) {
  Type elementType = args[0].getType();
  // TODO: Add support for integer types.
  assert(elementType.isa<::mlir::FloatType>() &&
         "Only support float case for relu");

  Value constZero = b.create<ConstantOp>(loc, FloatAttr::get(elementType, 0.0));
  Value pred = b.create<CmpFOp>(loc, CmpFPredicate::UGT, args[0], constZero);
  return b.create<SelectOp>(loc, pred, args[0], constZero);
}

namespace {

// Converts a unary op. There is no implicit broadcasting behavior, so these can
// be trivially lowered to linalg.
// TODO: For binary ops, we will need a "linalg.generic-like" op that models
// N-ary broadcasting and allows us to do multiversioning techniques for
// lowering to linalg. We can trivially handle this as through that
// abstraction instead.
struct ConvertUnaryOp : ConversionPattern {
  ConvertUnaryOp(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<AtenTanhOp>(op) && !isa<AtenReluOp>(op))
      return rewriter.notifyMatchFailure(op, "not a unary op");

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Value operand = operands[0];
    auto type = getTypeConverter()
                    ->convertType(op->getResult(0).getType())
                    .cast<RankedTensorType>();
    auto rank = type.getRank();

    SmallVector<StringRef> iteratorTypes(rank, "parallel");
    SmallVector<AffineMap> indexingMaps = {
        rewriter.getMultiDimIdentityMap(rank),
        rewriter.getMultiDimIdentityMap(rank)};

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, type, operand, operand,
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result;
          if (isa<AtenTanhOp>(op))
            result = b.create<math::TanhOp>(loc, args[0]);
          else if (isa<AtenReluOp>(op))
            result = createScalarRelu(b, loc, args);

          b.create<linalg::YieldOp>(loc, result);
        });

    return success();
  }
};
} // namespace

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchToLinalg
    : public ConvertTorchToLinalgBase<ConvertTorchToLinalg> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<StandardOpsDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect,
                           memref::MemRefDialect, math::MathDialect,
                           tensor::TensorDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    target.addIllegalOp<AtenMmOp>();
    patterns.add<ConvertAtenMmOp>(typeConverter, context);
    target.addIllegalOp<AtenLinearOp>();
    patterns.add<ConvertAtenLinearOp>(typeConverter, context);
    target.addIllegalOp<AtenTanhOp>();
    patterns.add<ConvertUnaryOp>(typeConverter, context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createConvertTorchToLinalgPass() {
  return std::make_unique<ConvertTorchToLinalg>();
}
