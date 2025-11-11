//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
namespace mlir::torch::TorchConversion {

#define GEN_PASS_DEF_CONVERTCUSTOMQUANTOP
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h.inc"

namespace {
class ConvertCustomQuantizedMatmulOp : public OpConversionPattern<OperatorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getName().str() != "quant.matmul_rhs_group_quant") {
      return failure();
    }
    Location loc = op->getLoc();
    if (failed(verifyLinalgCompatibleTypes(op, rewriter))) {
      return failure();
    }

    // get inputs: lhs, rhsQuant, scales, zps
    Value lhs = adaptor.getOperands()[0];
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    if (!lhsType) {
      return failure();
    }
    auto lhsShape = lhsType.getShape();
    int lhsReductDimSize = lhsShape.back();

    Value rhsQuant = adaptor.getOperands()[1];
    auto rhsType = cast<RankedTensorType>(rhsQuant.getType());
    if (!rhsType) {
      return failure();
    }
    auto rhsShape = rhsType.getShape();
    int rhsReductDimSize = rhsShape.back();
    Type rhsElementType = rhsType.getElementType();

    Value scales = adaptor.getOperands()[2];
    Value zps = adaptor.getOperands()[3];
    Value unpackedTypeWidth = adaptor.getOperands()[4];
    Value groupSize = adaptor.getOperands()[5];

    auto getConstantIntegerFromDefiningOp = [](Value operand,
                                               int &extractedInt) {
      auto castOp =
          dyn_cast<mlir::UnrealizedConversionCastOp>(operand.getDefiningOp());
      if (!castOp) {
        return failure();
      }
      auto constOp =
          dyn_cast<Torch::ConstantIntOp>(castOp.getOperand(0).getDefiningOp());
      if (!constOp) {
        return failure();
      }
      extractedInt = constOp.getValue();
      return success();
    };

    int gs;
    if (failed(getConstantIntegerFromDefiningOp(groupSize, gs))) {
      return failure();
    }
    int unpackedBitWidth;
    if (failed(getConstantIntegerFromDefiningOp(unpackedTypeWidth,
                                                unpackedBitWidth))) {
      return failure();
    }
    if (unpackedBitWidth !=
        static_cast<int>(rhsElementType.getIntOrFloatBitWidth())) {
      return failure();
    }

    // get outputs
    Type newResultType = getTypeConverter()->convertType(op.getType(0));
    auto resultType = cast<RankedTensorType>(newResultType);
    if (!resultType) {
      return failure();
    }
    auto resultShape = resultType.getShape();
    Type elementType = resultType.getElementType();

    // expand lhs
    std::vector<int64_t> lhsExpandedShape = {lhsShape[0], lhsShape[1],
                                             lhsReductDimSize / gs, gs};
    RankedTensorType lhsExpandedType =
        RankedTensorType::get(lhsExpandedShape, elementType);
    SmallVector<ReassociationIndices, 4> lhsReassociation = {{0}, {1}, {2, 3}};
    Value lhsExpanded = tensor::ExpandShapeOp::create(
        rewriter, loc, lhsExpandedType, lhs, lhsReassociation);

    // expand rhs
    std::vector<int64_t> rhsExpandedShape = {rhsShape[0], rhsReductDimSize / gs,
                                             gs};
    RankedTensorType rhsExpandedType =
        RankedTensorType::get(rhsExpandedShape, rhsElementType);
    SmallVector<ReassociationIndices, 4> rhsReassociation = {{0}, {1, 2}};
    Value rhsExpanded = tensor::ExpandShapeOp::create(
        rewriter, loc, rhsExpandedType, rhsQuant, rhsReassociation);
    Value cst0 = arith::ConstantOp::create(rewriter, loc,
                                           FloatAttr::get(elementType, 0.0));

    Value emptyDequant =
        tensor::EmptyOp::create(rewriter, loc, rhsExpandedShape, elementType);
    SmallVector<Value> dynDims;
    for (int i = 0; i < lhsType.getRank(); i++) {
      if (lhsType.isDynamicDim(i)) {
        dynDims.push_back(tensor::DimOp::create(rewriter, loc, lhs, i));
      }
    }
    Value empty = tensor::EmptyOp::create(rewriter, loc, resultShape,
                                          elementType, dynDims);
    Value output =
        linalg::FillOp::create(rewriter, loc, cst0, empty).getResult(0);

    AffineExpr d0, d1, d2, d3, d4;
    bindDims(getContext(), d0, d1, d2, d3, d4);
    auto c0 = rewriter.getAffineConstantExpr(0);
    auto map = AffineMap::get(3, 0, {d0, d1, d2}, rewriter.getContext());
    auto map1 = AffineMap::get(3, 0, {d0, d1, c0}, rewriter.getContext());
    auto map2 = AffineMap::get(5, 0, {d0, d1, d3, d4}, rewriter.getContext());
    auto map3 = AffineMap::get(5, 0, {d2, d3, d4}, rewriter.getContext());
    auto map4 = AffineMap::get(5, 0, {d0, d1, d2}, rewriter.getContext());
    SmallVector<AffineMap, 4> dqIndexingMaps = {map, map1, map1, map};
    SmallVector<AffineMap, 4> matIndexingMaps = {map2, map3, map4};

    SmallVector<utils::IteratorType> dequantIteratorTypes(
        3, utils::IteratorType::parallel);
    SmallVector<utils::IteratorType> matmulIteratorTypes = {
        utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::parallel, utils::IteratorType::reduction,
        utils::IteratorType::reduction};

    Value rhsDequant =
        linalg::GenericOp::create(
            rewriter, loc, emptyDequant.getType(),
            ValueRange{rhsExpanded, scales, zps}, emptyDequant,
            /*indexingMaps=*/dqIndexingMaps,
            /*iteratorTypes=*/dequantIteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value w = args[0], scale = args[1], zeroPoint = args[2];
              Value extw =
                  arith::ExtUIOp::create(b, loc, rewriter.getI32Type(), w);
              Value fp_extw =
                  arith::UIToFPOp::create(b, loc, rewriter.getF16Type(), extw);
              Value shifted = arith::SubFOp::create(b, loc, fp_extw, zeroPoint);
              Value dqw = arith::MulFOp::create(b, loc, shifted, scale);
              linalg::YieldOp::create(b, loc, dqw);
            })
            .getResult(0);

    Value matmulDequant = linalg::GenericOp::create(
                              rewriter, loc, output.getType(),
                              ValueRange{lhsExpanded, rhsDequant}, output,
                              /*indexingMaps=*/matIndexingMaps,
                              /*iteratorTypes=*/matmulIteratorTypes,
                              [&](OpBuilder &b, Location loc, ValueRange args) {
                                Value l = args[0], r = args[1], out = args[2];
                                Value pd = arith::MulFOp::create(b, loc, l, r);
                                Value ac =
                                    arith::AddFOp::create(b, loc, pd, out);
                                linalg::YieldOp::create(b, loc, ac);
                              })
                              .getResult(0);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, matmulDequant);
    return success();
  }
};
} // namespace

namespace {
class ConvertCustomQuantOpPass
    : public impl::ConvertCustomQuantOpBase<ConvertCustomQuantOpPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<Torch::TorchDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, func::FuncDialect,
                           tensor::TensorDialect, arith::ArithDialect,
                           Torch::TorchDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    target.addIllegalOp<OperatorOp>();
    patterns.add<ConvertCustomQuantizedMatmulOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createConvertCustomQuantOpPass() {
  return std::make_unique<ConvertCustomQuantOpPass>();
}

} // namespace mlir::torch::TorchConversion
