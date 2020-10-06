//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "npcomp/Dialect/RefBackend/IR/RefBackendDialect.h"
#include "npcomp/Dialect/RefBackend/IR/RefBackendOps.h"
#include "npcomp/Dialect/TCP/IR/TCPOps.h"
#include "npcomp/E2E/E2E.h"

using namespace mlir;
using namespace mlir::NPCOMP;

// TODO: Don't just open-code all shape transfer functions here.
static SmallVector<Value, 6> bypassResultShapes(Operation &op) {
  OpBuilder builder(&op);

  if (auto broadcastTo = dyn_cast<tcp::BroadcastToOp>(op)) {
    return {broadcastTo.shape()};
  }

  // Elementwise ops.
  if (isa<tcp::AddOp, tcp::MaxOp, tcp::ExpOp, tcp::TanhOp>(op)) {
    return {builder.create<shape::ShapeOfOp>(op.getLoc(), op.getOperand(0))};
  }

  if (auto matmul = dyn_cast<tcp::MatmulOp>(op)) {
    auto lhsRows = builder.create<DimOp>(op.getLoc(), matmul.lhs(), 0);
    auto rhsCols = builder.create<DimOp>(op.getLoc(), matmul.rhs(), 1);
    auto shape = builder.create<TensorFromElementsOp>(
        op.getLoc(), ValueRange({lhsRows, rhsCols}));
    return {shape};
  }

  // No shape transfer function.
  return {};
}

namespace {
// TODO: There is a coupling between this pass and LowerShapedResults.
// Any op that is wrapped in refback.shaped_results here needs to be known how
// to be lowered by LowerShapedResults.
class BypassShapes : public BypassShapesBase<BypassShapes> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<shape::ShapeDialect, refback::RefBackendDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    func.walk([&](Operation *opPtr) {
      Operation &op = *opPtr;
      SmallVector<Value, 6> resultShapes = bypassResultShapes(op);
      if (resultShapes.empty())
        return;
      // We have result shapes, so wrap this op in a refback.shaped_results op.
      OpBuilder builder(&op);
      auto shapedResults = builder.create<refback::ShapedResultsOp>(
          op.getLoc(), op.getResultTypes(), resultShapes);
      op.replaceAllUsesWith(shapedResults);

      // Move the op into the body and yield the results.
      Block *body = builder.createBlock(&shapedResults.body());
      op.moveBefore(body, body->end());
      builder.create<refback::YieldOp>(op.getLoc(), op.getResults());
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::NPCOMP::createBypassShapesPass() {
  return std::make_unique<BypassShapes>();
}
