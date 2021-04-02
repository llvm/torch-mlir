//===- NumpyOps.cpp - Core numpy dialect ops --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Numpy/IR/NumpyOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Numpy;

//----------------------------------------------------------------------------//
// Type inference
//----------------------------------------------------------------------------//

/// Adds constraints to relating a unary op that accepts and returns either
/// tensor or ndarray types where the dtype should be the same.
/// Type constraints are added on the dtype, not the outer object type.
static void constrainUnaryDtypeInvariantOp(Typing::CPA::Context &context,
                                           Value source, Value dest,
                                           Operation *op) {
  auto &env = context.getCurrentEnvironment();
  auto *sourceTn =
      llvm::dyn_cast<Typing::CPA::ObjectValueType>(env.mapValueToType(source));
  auto *destTn =
      llvm::dyn_cast<Typing::CPA::ObjectValueType>(env.mapValueToType(dest));
  if (sourceTn && destTn && sourceTn->getFieldCount() == 1 &&
      destTn->getFieldCount() == 1) {
    context.getConstraint(sourceTn->getFieldTypes().front(),
                          destTn->getFieldTypes().front());
  }
}

void CreateArrayFromTensorOp::addCPAConstraints(Typing::CPA::Context &context) {
  constrainUnaryDtypeInvariantOp(context, source(), dest(), *this);
}

void CopyToTensorOp::addCPAConstraints(Typing::CPA::Context &context) {
  constrainUnaryDtypeInvariantOp(context, source(), dest(), *this);
}

void BuiltinUfuncCallOp::addCPAConstraints(Typing::CPA::Context &context) {
  // TODO: This should really be a function call chosen so as to promote
  // arguments. For now, though, we just say that the result is constrained
  // to the inputs. Note that not all ufuncs transfer types like this.
  // We just pretend this is two unary functions that write into the output.
  for (auto input : inputs()) {
    constrainUnaryDtypeInvariantOp(context, input, output(), *this);
  }
}

//----------------------------------------------------------------------------//
// StaticInfoCast
//----------------------------------------------------------------------------//

bool StaticInfoCastOp::areCastCompatible(mlir::TypeRange inputs,
                                         mlir::TypeRange outputs) {
  auto input = inputs[0].cast<NdArrayType>();
  auto output = outputs[0].cast<NdArrayType>();
  if (input.getOptionalShape() && output.getOptionalShape()) {
    if (failed(verifyCompatibleShape(*input.getOptionalShape(),
                                     *output.getOptionalShape())))
      return false;
  }
  return input.getDtype() == output.getDtype() ||
         input.getDtype().isa<AnyDtypeType>() ||
         output.getDtype().isa<AnyDtypeType>();
}

void StaticInfoCastOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *context) {
  // static_info_cast(oneUse@create_array_from_tensor(%tensor))
  // -->
  // create_array_from_tensor(tensor_static_info_cast(%tensor))
  //
  // This pattern tends to create more tensor code and less array code.
  // This form is considered more canonical because it has same number of ops
  // but is more analyzable.
  //
  // TODO: Consider a world where we numpy.ndarray can track an "immutable" bit
  // which makes it tensor-like. Is that useful?
  patterns.add(+[](StaticInfoCastOp op, PatternRewriter &rewriter) {
    auto createArray = op.getOperand().getDefiningOp<CreateArrayFromTensorOp>();
    if (!createArray || !createArray.getResult().hasOneUse())
      return failure();
    auto tensorCast = rewriter.create<TensorStaticInfoCastOp>(
        op.getLoc(), op.getType().cast<NdArrayType>().toTensorType(),
        createArray.getOperand());
    rewriter.replaceOpWithNewOp<CreateArrayFromTensorOp>(op, op.getType(),
                                                         tensorCast);
    rewriter.eraseOp(createArray);
    return success();
  });
}

//----------------------------------------------------------------------------//
// TensorStaticInfoCast
//----------------------------------------------------------------------------//

bool TensorStaticInfoCastOp::areCastCompatible(mlir::TypeRange inputs,
                                               mlir::TypeRange outputs) {
  auto input = inputs[0].cast<TensorType>();
  auto output = outputs[0].cast<TensorType>();
  if (input.hasRank() && output.hasRank()) {
    if (failed(verifyCompatibleShape(input.getShape(), output.getShape())))
      return false;
  }
  return input.getElementType() == output.getElementType() ||
         input.getElementType().isa<AnyDtypeType>() ||
         output.getElementType().isa<AnyDtypeType>();
}

//----------------------------------------------------------------------------//
// CreateArrayFromTensorOp
//----------------------------------------------------------------------------//

namespace {
/// Match create_array_from_tensor -> copy_to_tensor and elide in favor
/// of the original tensor.
class ElideCreateRedundantArrayFromTensor
    : public OpRewritePattern<CopyToTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CopyToTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto createArrayOp =
        dyn_cast_or_null<CreateArrayFromTensorOp>(op.source().getDefiningOp());
    if (createArrayOp && createArrayOp.dest().hasOneUse()) {
      rewriter.replaceOp(op, createArrayOp.source());
    }
    return success();
  }
};
} // namespace

void CopyToTensorOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *context) {
  patterns.add<ElideCreateRedundantArrayFromTensor>(context);
}

#define GET_OP_CLASSES
#include "npcomp/Dialect/Numpy/IR/NumpyOps.cpp.inc"
