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
#include "npcomp/Dialect/TCP/IR/TCPOps.h"
#include "npcomp/E2E/E2E.h"

using namespace mlir;
using namespace mlir::NPCOMP;

static bool isSimpleElementwiseLinalgGeneric(linalg::GenericOp op) {
  // Only handle generic ops where all operands and results are tensors.
  if (!llvm::all_of(op.getOperandTypes(),
                    [](Type type) { return type.isa<RankedTensorType>(); })) {
    return false;
  }
  if (!llvm::all_of(op.getResultTypes(),
                    [](Type type) { return type.isa<RankedTensorType>(); })) {
    return false;
  }

  // TODO: Loosen restrictions on indexing maps.
  // This will require more principled handling of shape reification
  // earlier in the compilation stack, as in general output shapes of a
  // linalg.generic cannot be inferred easily.
  // See:
  // https://llvm.discourse.group/t/computing-output-shapes-of-structured-ops-on-tensors/866
  if (!llvm::all_of(op.indexing_maps(), [](Attribute map) {
        return map.cast<AffineMapAttr>().getValue().isIdentity();
      })) {
    return false;
  }
  if (!llvm::all_of(op.iterator_types(), [](Attribute str) {
        return str.cast<StringAttr>().getValue() ==
               getParallelIteratorTypeName();
      })) {
    return false;
  }

  return true;
}

// TODO: Don't just open-code all shape transfer functions here.
// Note: for now, we can't just rely on an OpInterface, since OpInterfaces
// cannot be "externally applied". E.g. we can't change the definition of
// linalg::GenericOp.
static SmallVector<Value, 6> bypassResultShapes(Operation &op) {
  OpBuilder builder(&op);
  if (auto linalgGeneric = dyn_cast<linalg::GenericOp>(op)) {
    // TODO: Avoid this excessive restriction.
    // This will require more principled handling of the lowering to
    // linalg.generic -- it should generally happen after this pass, becaue in
    // general output shapes of a linalg.generic cannot be inferred easily. See:
    // https://llvm.discourse.group/t/computing-output-shapes-of-structured-ops-on-tensors/866
    if (!isSimpleElementwiseLinalgGeneric(linalgGeneric))
      return {};
    // All shapes of all operands and results are the same for now. So
    // arbitrarily pick the first operand.
    return {builder.create<shape::ShapeOfOp>(op.getLoc(), op.getOperand(0))};
  }

  if (auto broadcastTo = dyn_cast<tcp::BroadcastToOp>(op)) {
    return {broadcastTo.shape()};
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
// Any op that is wrapped in tcp.shaped_results here needs to be known how to be
// lowered by LowerShapedResults.
class BypassShapes : public BypassShapesBase<BypassShapes> {
  void runOnOperation() {
    auto func = getOperation();
    func.walk([&](Operation *opPtr) {
      Operation &op = *opPtr;
      SmallVector<Value, 6> resultShapes = bypassResultShapes(op);
      if (resultShapes.empty())
        return;
      // We have result shapes, so wrap this op in a tcp.shaped_results op.
      OpBuilder builder(&op);
      auto shapedResults = builder.create<tcp::ShapedResultsOp>(
          op.getLoc(), op.getResultTypes(), resultShapes);
      op.replaceAllUsesWith(shapedResults);

      // Move the op into the body and yield the results.
      Block *body = builder.createBlock(&shapedResults.body());
      op.moveBefore(body, body->end());
      builder.create<tcp::YieldOp>(op.getLoc(), op.getResults());
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::NPCOMP::createBypassShapesPass() {
  return std::make_unique<BypassShapes>();
}
