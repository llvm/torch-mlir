//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "npcomp/Conversion/BasicpyToStd/Patterns.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

namespace {

bool isLegalBinaryOpType(Type type) {
  if (type.isIntOrFloat()) {
    return type.getIntOrFloatBitWidth() > 1; // Do not match i1
  }

  return false;
}

// Convert to std ops when all types match. It is assumed that additional
// patterns and type inference are used to get into this form.
class NumericBinaryExpr : public OpRewritePattern<Basicpy::BinaryExprOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Basicpy::BinaryExprOp op,
                                PatternRewriter &rewriter) const override {
    // Match failure unless if both:
    //   a) operands/results are the same type
    //   b) matches a set of supported primitive types
    //   c) the operation maps to a simple std op without further massaging
    auto valueType = op.left().getType();
    if (valueType != op.right().getType() || valueType != op.result().getType())
      return failure();
    if (!isLegalBinaryOpType(valueType))
      return failure();

    auto operation = Basicpy::symbolizeBinaryOperation(op.operation());
    if (!operation)
      return failure();
    auto left = op.left();
    auto right = op.right();

    // Generally, int and float ops in std are different.
    using Basicpy::BinaryOperation;
    if (valueType.isa<IntegerType>()) {
      // Note that not all operations make sense or are defined for integer
      // math. Of specific note is the Div vs FloorDiv distinction.
      switch (*operation) {
      case BinaryOperation::Add:
        rewriter.replaceOpWithNewOp<AddIOp>(op, left, right);
        return success();
      case BinaryOperation::BitAnd:
        rewriter.replaceOpWithNewOp<AndOp>(op, left, right);
        return success();
      case BinaryOperation::BitOr:
        rewriter.replaceOpWithNewOp<OrOp>(op, left, right);
        return success();
      case BinaryOperation::BitXor:
        rewriter.replaceOpWithNewOp<XOrOp>(op, left, right);
        return success();
      case BinaryOperation::FloorDiv:
        // TODO: This is not a precise match for negative division.
        // SignedDivIOp rounds towards zero and python rounds towards
        // most negative.
        rewriter.replaceOpWithNewOp<SignedDivIOp>(op, left, right);
        return success();
      case BinaryOperation::LShift:
        rewriter.replaceOpWithNewOp<ShiftLeftOp>(op, left, right);
        return success();
      case BinaryOperation::Mod:
        rewriter.replaceOpWithNewOp<SignedRemIOp>(op, left, right);
        return success();
      case BinaryOperation::Mult:
        rewriter.replaceOpWithNewOp<MulIOp>(op, left, right);
        return success();
      case BinaryOperation::RShift:
        rewriter.replaceOpWithNewOp<SignedShiftRightOp>(op, left, right);
        return success();
      case BinaryOperation::Sub:
        rewriter.replaceOpWithNewOp<SubIOp>(op, left, right);
        return success();
      default:
        return failure();
      }
    } else if (valueType.isa<FloatType>()) {
      // Note that most operations are not supported on floating point values.
      // In addition, some cannot be directly implemented with single std
      // ops.
      switch (*operation) {
      case BinaryOperation::Add:
        rewriter.replaceOpWithNewOp<AddFOp>(op, left, right);
        return success();
      case BinaryOperation::Div:
        rewriter.replaceOpWithNewOp<DivFOp>(op, left, right);
        return success();
      case BinaryOperation::FloorDiv:
        // TODO: Implement floating point floor division.
        return rewriter.notifyMatchFailure(
            op, "floating point floor division not implemented");
      case BinaryOperation::Mod:
        // TODO: Implement floating point mod.
        return rewriter.notifyMatchFailure(
            op, "floating point mod not implemented");
      case BinaryOperation::Mult:
        rewriter.replaceOpWithNewOp<MulFOp>(op, left, right);
        return success();
      case BinaryOperation::Sub:
        rewriter.replaceOpWithNewOp<SubFOp>(op, left, right);
        return success();
      default:
        return failure();
      }
    }

    return failure();
  }
};

Optional<CmpIPredicate>
mapBasicpyPredicateToCmpI(Basicpy::CompareOperation predicate) {
  using Basicpy::CompareOperation;
  switch (predicate) {
  case CompareOperation::Eq:
    return CmpIPredicate::eq;
  case CompareOperation::Gt:
    return CmpIPredicate::sgt;
  case CompareOperation::GtE:
    return CmpIPredicate::sge;
  case CompareOperation::Is:
    return CmpIPredicate::eq;
  case CompareOperation::IsNot:
    return CmpIPredicate::ne;
  case CompareOperation::Lt:
    return CmpIPredicate::slt;
  case CompareOperation::LtE:
    return CmpIPredicate::sle;
  case CompareOperation::NotEq:
    return CmpIPredicate::ne;
  default:
    return llvm::None;
  }
}

Optional<CmpFPredicate>
mapBasicpyPredicateToCmpF(Basicpy::CompareOperation predicate) {
  using Basicpy::CompareOperation;
  switch (predicate) {
  case CompareOperation::Eq:
    return CmpFPredicate::OEQ;
  case CompareOperation::Gt:
    return CmpFPredicate::OGT;
  case CompareOperation::GtE:
    return CmpFPredicate::OGE;
  case CompareOperation::Is:
    return CmpFPredicate::OEQ;
  case CompareOperation::IsNot:
    return CmpFPredicate::ONE;
  case CompareOperation::Lt:
    return CmpFPredicate::OLT;
  case CompareOperation::LtE:
    return CmpFPredicate::OLE;
  case CompareOperation::NotEq:
    return CmpFPredicate::ONE;
  default:
    return llvm::None;
  }
}

class NumericCompare : public OpRewritePattern<Basicpy::BinaryCompareOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Basicpy::BinaryCompareOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto valueType = op.left().getType();
    if (valueType != op.right().getType())
      return failure();
    if (!isLegalBinaryOpType(valueType))
      return failure();
    auto bpyPredicate = Basicpy::symbolizeCompareOperation(op.operation());
    if (!bpyPredicate)
      return failure();

    if (valueType.isa<IntegerType>()) {
      if (auto stdPredicate = mapBasicpyPredicateToCmpI(*bpyPredicate)) {
        auto cmp =
            rewriter.create<CmpIOp>(loc, *stdPredicate, op.left(), op.right());
        rewriter.replaceOpWithNewOp<Basicpy::BoolCastOp>(
            op, Basicpy::BoolType::get(rewriter.getContext()), cmp);
        return success();
      } else {
        return rewriter.notifyMatchFailure(op, "unsupported compare operation");
      }
    } else if (valueType.isa<FloatType>()) {
      if (auto stdPredicate = mapBasicpyPredicateToCmpF(*bpyPredicate)) {
        auto cmp =
            rewriter.create<CmpFOp>(loc, *stdPredicate, op.left(), op.right());
        rewriter.replaceOpWithNewOp<Basicpy::BoolCastOp>(
            op, Basicpy::BoolType::get(rewriter.getContext()), cmp);
        return success();
      } else {
        return rewriter.notifyMatchFailure(op, "unsupported compare operation");
      }
    }

    return failure();
  }
};

// Converts the as_i1 op for numeric types.
class NumericToI1 : public OpRewritePattern<Basicpy::AsI1Op> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Basicpy::AsI1Op op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto operandType = op.operand().getType();
    if (operandType.isa<IntegerType>()) {
      auto zero = rewriter.create<ConstantIntOp>(loc, 0, operandType);
      rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::ne, op.operand(),
                                          zero);
      return success();
    } else if (operandType.isa<FloatType>()) {
      auto zero = rewriter.create<ConstantOp>(loc, operandType,
                                              FloatAttr::get(operandType, 0.0));
      rewriter.replaceOpWithNewOp<CmpFOp>(op, CmpFPredicate::ONE, op.operand(),
                                          zero);
      return success();
    }
    return failure();
  }
};

} // namespace

void mlir::NPCOMP::populateBasicpyToStdPrimitiveOpPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<NumericBinaryExpr>(context);
  patterns.insert<NumericCompare>(context);
  patterns.insert<NumericToI1>(context);
}
