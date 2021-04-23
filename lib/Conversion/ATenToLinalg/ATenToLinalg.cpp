//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Conversion/ATenToLinalg/ATenToLinalg.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h" // TODO: For `memref.dim`.
#include "mlir/Dialect/Traits.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "npcomp/Dialect/ATen/IR/ATenDialect.h"

using namespace mlir;
using namespace mlir::NPCOMP;

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

static LogicalResult verifyLinalgCompatibleTypes(Operation *op, PatternRewriter &rewriter) {
  // For now, use a small allowlist of types we don't reject.
  // The main culprit in practice is that !numpy.any_dtype might be present
  // if shape/dtype inference wasn't good enough.
  auto isValidLinalgType = [](Type type) {
    if (auto rankedTensor = type.dyn_cast<RankedTensorType>()) {
      if (BaseMemRefType::isValidElementType(rankedTensor.getElementType()))
        return true;
    }
    if (type.isa<FloatType, IntegerType, IndexType>())
      return true;
    return false;
  };
  bool valid = llvm::all_of(op->getOperandTypes(), isValidLinalgType) &&
               llvm::all_of(op->getResultTypes(), isValidLinalgType);
  if (!valid)
    return rewriter.notifyMatchFailure(op, "type cannot be lowered to linalg");
  return success();
}

LogicalResult convertMmOp(aten::MmOp op, PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  Value lhs = op.getOperand(0);
  Value rhs = op.getOperand(1);

  // A user can write an errorneous program where `aten.mm` is in fact called
  // with operands of invalid rank or dtype. We cannot convert to linalg in this
  // case or we will get a verifier error, which corresponds to breaking of
  // *internal* compiler invariants, and for a user manifests as a compiler
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
      rewriter.getStringAttr("mismatching contracting dimension for aten.mm"));

  Type elementType = op.getType().cast<TensorType>().getElementType();
  Value initTensor = rewriter.create<linalg::InitTensorOp>(
      loc, ValueRange{lhsDim0, rhsDim1}, elementType);
  Value c0 = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0.0));
  Value zeroFill =
      rewriter.create<linalg::FillOp>(loc, initTensor, c0).getResult(0);
  Value matmul = rewriter
                     .create<linalg::MatmulOp>(loc, zeroFill.getType(),
                                               ValueRange{lhs, rhs}, zeroFill)
                     .getResult(0);
  // When constructed with just dynamic sizes, InitTensorOp will have a result
  // type which has all `?`'s for dimensions, which might not be the result
  // type of `op`. The constraints on later linalg ops means that the result of
  // the MatmulOp will have this type too. So cast it to the desired type so
  // that in the end we have the original result type.
  rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(), matmul);

  return success();
}

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------

namespace {
class ConvertATenToLinalg
    : public ConvertATenToLinalgBase<ConvertATenToLinalg> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    (void)applyPatternsAndFoldGreedily(getOperation(), getPatterns());
  }

  FrozenRewritePatternList getPatterns() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add(convertMmOp);
    return std::move(patterns);
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createConvertATenToLinalgPass() {
  return std::make_unique<ConvertATenToLinalg>();
}
