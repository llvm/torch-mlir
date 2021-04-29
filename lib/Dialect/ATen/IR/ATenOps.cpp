//===- ATenDialect.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/ATen/IR/ATenDialect.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"
#include "npcomp/Dialect/Torch/IR/TorchTypes.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::aten;

//===----------------------------------------------------------------------===//
// IsOp
//===----------------------------------------------------------------------===//

OpFoldResult IsOp::fold(ArrayRef<Attribute> operands) {
  auto lhsType = self().getType();
  auto rhsType = obj().getType();
  // If either type is a NoneType, make it be the lhsType.
  if (rhsType.isa<Basicpy::NoneType>())
    std::swap(lhsType, rhsType);
  // TODO: Implement and use subtype infra for this.
  // If neither type is a subtype of the other, then the result is false.
  if (lhsType.isa<Basicpy::NoneType>() && !rhsType.isa<Torch::OptionalType>())
    return IntegerAttr::get(IntegerType::get(getContext(), 1), 0);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// LenTOp
//===----------------------------------------------------------------------===//

void LenTOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                         MLIRContext *context) {
  patterns.add(+[](LenTOp op, PatternRewriter &rewriter) {
    auto buildList = op.getOperand().getDefiningOp<Basicpy::BuildListOp>();
    if (!buildList)
      return rewriter.notifyMatchFailure(op, "operand not basicpy.build_list");
    rewriter.replaceOpWithNewOp<::mlir::ConstantOp>(
        op, rewriter.getI64IntegerAttr(buildList.getNumOperands()));
    return success();
  });
}

//===----------------------------------------------------------------------===//
// SizeOp
//===----------------------------------------------------------------------===//

void SizeOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                         MLIRContext *context) {
  patterns.add(+[](SizeOp op, PatternRewriter &rewriter) {
    auto type = op.getOperand().getType().dyn_cast<RankedTensorType>();
    if (!type)
      return rewriter.notifyMatchFailure(op, "not a ranked tensor");
    SmallVector<Value> listElements;
    for (int64_t size : type.getShape()) {
      listElements.push_back(rewriter.create<::mlir::ConstantOp>(
          op->getLoc(), rewriter.getI64IntegerAttr(size)));
    }
    rewriter.replaceOpWithNewOp<Basicpy::BuildListOp>(
        op, Basicpy::ListType::get(rewriter.getContext()), listElements);
    return success();
  });
}

#define GET_OP_CLASSES
#include "npcomp/Dialect/ATen/IR/ATenOps.cpp.inc"

#include "npcomp/Dialect/ATen/IR/GeneratedATenOps.cpp.inc"
