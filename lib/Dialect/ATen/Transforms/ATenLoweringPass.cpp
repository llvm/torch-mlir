//===- ATenLoweringPass.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/ATen/Transforms/ATenLoweringPass.h"
#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "npcomp/Dialect/ATen/Transforms/ATenToStd.h"

#include "mlir/Dialect/Affine/EDSC/Builders.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/EDSC/Builders.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <sstream>

using namespace mlir;
using namespace edsc::intrinsics;
using namespace mlir::NPCOMP::aten;

using callOperation = edsc::OperationBuilder<mlir::CallOp>;
using call = edsc::ValueBuilder<mlir::CallOp>;
using constInt = edsc::intrinsics::std_constant_int;
using constFloat = edsc::intrinsics::std_constant_float;

namespace {

/// Utility function for type casting: this is making the type checker happy,
/// while delaying the actual work involved to convert the type. Most of the
/// time both side of the cast (producer and consumer) will be lowered to a
/// dialect like LLVM and end up with the same LLVM representation, at which
/// point this becomes a no-op and is eliminated.
static Value typeCast(PatternRewriter &builder, Value val, Type destTy) {
  if (val.getType() == destTy)
    return val;
  return builder
      .create<mlir::NPCOMP::aten::TypeCastOp>(val.getLoc(), destTy, val)
      .getResult();
}

/// Given a MemRefType, return a new MemRefType with the same rank, but
/// unknown shape.
static MemRefType getShapeErasedMemRefType(MemRefType type) {
  std::vector<int64_t> shape = type.getShape();
  for (size_t i = 0, e = shape.size(); i < e; i++) {
    shape[i] = -1;
  }
  return MemRefType::get(shape, type.getElementType(), type.getAffineMaps(),
                         type.getMemorySpace());
}

/// Create a type cast to memref
static Value memRefTypeCast(PatternRewriter &builder, Value val) {
  Type type = val.getType();

  if (auto memrefTy = type.dyn_cast<MemRefType>()) {
    MemRefType newType = getShapeErasedMemRefType(memrefTy);
    return builder.create<MemRefCastOp>(val.getLoc(), val, newType).getResult();
  }
  if (auto tensorTy = type.dyn_cast<TensorType>()) {
    auto memRefType = mlir::MemRefType::get(tensorTy.getShape(),
                                            tensorTy.getElementType(), {}, 0);
    return typeCast(builder, val, memRefType);
  }
  return val;
}

// Mangle a type in a way that encodes the full shape information.
// TODO: Currently only supports MemRef, Float, Integer, and AtenList (poorly)
static std::string getFullyMangledType(const Type ty) {
  std::stringstream ret;

  if (const MemRefType mrt = ty.dyn_cast<const MemRefType>()) {
    ret << "M";
    auto shape = mrt.getShape();
    const Type elem = mrt.getElementType();
    for (auto s : shape)
      ret << s << "x";
    ret << getFullyMangledType(elem);
  } else if (FloatType ft = ty.dyn_cast<FloatType>()) {
    ret << "F" << ft.getWidth();
  } else if (const IntegerType it = ty.dyn_cast<const IntegerType>()) {
    ret << "I" << it.getWidth();
  } else if (const mlir::NPCOMP::aten::ATenListType alt =
                 ty.dyn_cast<const mlir::NPCOMP::aten::ATenListType>()) {

  } else {
    Type t = ty;
    t.dump();
    assert(0 && "unhandled type in getFullyMangledType");
  }
  return ret.str();
}

// Mangle the argument ranks into the function name.
// TODO: Currently only supports MemRef, Float, Integer, and AtenList (poorly)
static std::string getSimplyMangledType(const Type ty) {
  std::stringstream ret;

  if (const MemRefType mrt = ty.dyn_cast<const MemRefType>()) {
    //    ret << "M";
    ArrayRef<int64_t> shape = mrt.getShape();
    const Type elem = mrt.getElementType();
    ret << shape.size();
    ret << getFullyMangledType(elem);
  } else if (FloatType ft = ty.dyn_cast<FloatType>()) {
    // ret << "F" << ft.getWidth();
  } else if (const IntegerType it = ty.dyn_cast<const IntegerType>()) {
    // ret << "I" << it.getWidth();
  } else if (const mlir::NPCOMP::aten::ATenListType alt =
                 ty.dyn_cast<const mlir::NPCOMP::aten::ATenListType>()) {

  } else {
    Type t = ty;
    t.dump();
    assert(0 && "unhandled type in getSimplyMangledType");
  }
  return ret.str();
}

// Return a simply mangled function name.  The function name is constructed
// from the prefix, the mangled result types, the mangled operand types.
// Types are mangled in a way that encodes only the rank.  Shape information
// is passed runtime using the standard calling convention.  This simpler
// version of mangling allows us to implement most of the functions with only
// a few variations. However, it means we need to convert from tensor types
// with known size to tensor types with unknown size to have a consistent
// runtime calling convention.
static std::string getSimplyMangledFuncName(std::string prefix,
                                            ArrayRef<Type> operTy,
                                            ArrayRef<Type> resultTy) {
  std::string sep = "_";

  std::string ret = prefix;
  for (const Type t : resultTy)
    ret = ret + sep + getSimplyMangledType(t);
  for (const Type t : operTy) {
    std::string s = getSimplyMangledType(t);
    if (s.size() > 0)
      ret = ret + sep + getSimplyMangledType(t);
  }
  ret += "_out";

  return ret;
}

std::string getMangledFuncName(std::string prefix, ArrayRef<Type> opTys,
                               ArrayRef<Type> retTys) {
  return getSimplyMangledFuncName(prefix, opTys, retTys);
}

static FuncOp getATenFn(ModuleOp module, std::string mangledFunctionName,
                        ArrayRef<Value> operands, ArrayRef<Type> retTys) {
  Builder builder(module);

  SmallVector<Type, 8> tys;
  for (Value o : operands) {
    Type t = o.getType();
    // Erase the dimensions of the memref.
    if (t.isa<MemRefType>()) {
      auto mt = t.cast<MemRefType>();
      tys.push_back(getShapeErasedMemRefType(mt));
    } else
      tys.push_back(t);
  }

  auto fnTy = builder.getFunctionType(tys, retTys);

  auto fn = module.lookupSymbol<FuncOp>(mangledFunctionName);

  if (!fn) {
    fn = FuncOp::create(builder.getUnknownLoc(), mangledFunctionName, fnTy);
    module.push_back(fn);
  }

  return fn;
}

/// Lower an aten.add to an affine loop nest.
class AddOpConversion_affine : public ConversionPattern {
public:
  explicit AddOpConversion_affine(MLIRContext *context)
      : ConversionPattern(mlir::NPCOMP::aten::AddOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto add = cast<mlir::NPCOMP::aten::AddOp>(op);
    auto loc = add.getLoc();
    Type resultTy = add.getResult().getType();
    TensorType tensorResultTy = resultTy.cast<TensorType>();
    MemRefType memRefResultTy = mlir::MemRefType::get(
        tensorResultTy.getShape(), tensorResultTy.getElementType(), {}, 0);

    Value result = rewriter.create<AllocOp>(loc, memRefResultTy);
    Value lhs = memRefTypeCast(rewriter, operands[0]);
    Value rhs = memRefTypeCast(rewriter, operands[1]);
    using namespace edsc;

    ScopedContext scope(rewriter, loc);
    Value zero = intrinsics::std_constant_index(0);
    MemRefBoundsCapture vRes(result), vLHS(lhs), vRHS(rhs);
    StdIndexedValue iRes(result), iLHS(lhs), iRHS(rhs);
    Value M(vRes.ub(0));
    if (vRes.rank() == 1) {
      affineLoopNestBuilder({zero}, {M}, 1, [&](ValueRange ivs) {
        Value i = ivs[0];
        iRes(i) = iLHS(i) + iRHS(i);
      });
    } else if (vRes.rank() == 2) {
      Value N(vRes.ub(1));
      affineLoopNestBuilder({zero, zero}, {M, N}, {1, 1}, [&](ValueRange ivs) {
        Value i = ivs[0];
        Value j = ivs[1];
        iRes(i, j) = iLHS(i, j) + iRHS(i, j);
      });
    } else if (vRes.rank() == 3) {
      Value N(vRes.ub(1));
      Value O(vRes.ub(2));
      affineLoopNestBuilder({zero, zero, zero}, {M, N, O}, {1, 1, 1},
                            [&](ValueRange ivs) {
                              Value i = ivs[0];
                              Value j = ivs[1];
                              Value k = ivs[2];
                              iRes(i, j, k) = iLHS(i, j, k) + iRHS(i, j, k);
                            });
    } else {
      Value N(vRes.ub(1));
      Value O(vRes.ub(2));
      Value P(vRes.ub(3));
      affineLoopNestBuilder({zero, zero, zero, zero}, {M, N, O, P},
                            {1, 1, 1, 1}, [&](ValueRange ivs) {
                              Value i = ivs[0];
                              Value j = ivs[1];
                              Value k = ivs[2];
                              Value l = ivs[3];
                              iRes(i, j, k, l) =
                                  iLHS(i, j, k, l) + iRHS(i, j, k, l);
                            });
    }
    // Return the newly allocated buffer.
    rewriter.replaceOp(op, {result});
    return success();
  }
};

// Replace the given operation with a call to the given function.
// The function is assumed to accept memrefs and scalar types and return
// Memrefs. Here the result types are converted back to the result types of op,
// but operands are NOT converted.  This allows non-standard mappings from
// operand types to function types.
LogicalResult rewriteWithVoidFunctionCallExplicit(
    Operation *op, ArrayRef<Value> callops, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter, std::string functionName) {

  auto loc = op->getLoc();
  edsc::ScopedContext scope(rewriter, loc);

  // The original operation types.
  SmallVector<Type, 8> opTys;
  // Shape erased versions of the original operation types.
  SmallVector<Type, 8> erasedOpTys;
  for (const Value &o : callops) {
    Type t = o.getType();
    opTys.push_back(t);
    if (t.isa<MemRefType>())
      erasedOpTys.push_back(getShapeErasedMemRefType(t.cast<MemRefType>()));
    else
      erasedOpTys.push_back(t);
  }

  std::vector<Value> newOps = callops;
  SmallVector<Value, 8> newResults;

  // Result types of the original operation, converted to memrefs.
  SmallVector<Type, 8> retTys;
  // Erased version of the return type.  This is the return types of the
  // generated function call.
  SmallVector<Type, 8> erasedRetTys;
  for (const auto &o : op->getResults()) {
    Type t = o.getType();
    if (t.isa<TensorType>()) {
      TensorType tensorResultTy = t.cast<TensorType>();
      MemRefType memRefResultTy = mlir::MemRefType::get(
          tensorResultTy.getShape(), tensorResultTy.getElementType(), {}, 0);
      retTys.push_back(memRefResultTy);

      // assume memRefResultTy has known shape, so we don't need any
      // dynamic dimensions for the alloc.
      assert(memRefResultTy.hasStaticShape());
      Value allocVal = rewriter.create<AllocOp>(op->getLoc(), memRefResultTy);
      Value castVal = memRefTypeCast(rewriter, allocVal);
      newOps.push_back(castVal);
      newResults.push_back(allocVal);
    } else {
      return failure();
    }
  }

  SmallVector<Type, 8> empty;
  std::string mangledFunctionName =
      getMangledFuncName(functionName, opTys, retTys);
  FuncOp funcOp = getATenFn(op->getParentOfType<ModuleOp>(),
                            mangledFunctionName, newOps, empty);

  callOperation(empty, rewriter.getSymbolRefAttr(funcOp), newOps);

  rewriter.replaceOp(op, newResults);
  return success();
}

// Replace the given operation with a call to the given function.
// The function is assumed to accept memrefs and scalar types and return
// Memrefs.  Other operand types (e.g. aten.list and tensor<> are converted
// appropriately.  The called function passes results of the original function
// as memref arguments at the end of the original set of operands.
LogicalResult rewriteWithFunctionCall(Operation *op, ArrayRef<Value> operands,
                                      ConversionPatternRewriter &rewriter,
                                      std::string functionName) {
  auto loc = op->getLoc();
  edsc::ScopedContext scope(rewriter, loc);

  // Convert the arguments to the original call.
  SmallVector<Value, 8> callops;
  for (auto &o : operands) {
    Type t = o.getType();
    if (t.isa<MemRefType>()) {
      // Cast it to some memref type that we accept
      callops.push_back(memRefTypeCast(rewriter, o));
    } else if (t.isa<IntegerType>() || t.isa<FloatType>()) {
      callops.push_back(o);
    } else if (t.isa<ATenListType>()) {
      // FIXME: lots of assumptions here.
      auto unpack = [](auto &op, auto &v) -> void {
        auto co = cast<mlir::NPCOMP::aten::ConstantOp>(op.getDefiningOp());
        DenseElementsAttr a =
            co.template getAttrOfType<DenseElementsAttr>("value");
        for (auto i : a.getIntValues())
          v.push_back(i.getSExtValue());
      };
      std::vector<uint64_t> values;
      unpack(o, values);
      callops.push_back(constInt(values[0], 32));
    } else {
      return failure();
    }
  }
  return rewriteWithVoidFunctionCallExplicit(op, callops, operands, rewriter,
                                             functionName);
}

/// Lower Add
template <typename Op>
class ATenFunctionCallConversion : public ConversionPattern {
public:
  explicit ATenFunctionCallConversion(MLIRContext *context)
      : ConversionPattern(Op::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter,
                                   Op::getFunctionConversionName());
  }
};

/// Lower aten.constant
class ConstantOpConversion : public ConversionPattern {
public:
  explicit ConstantOpConversion(MLIRContext *context)
      : ConversionPattern(mlir::NPCOMP::aten::ConstantOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    Value result = op->getResult(0);
    Type t = result.getType();
    if (t.isa<IntegerType>()) {
      auto it = t.cast<IntegerType>();
      if (it.getWidth() > 1) {
        auto a = op->getAttrOfType<IntegerAttr>("value");
        SmallVector<Value, 8> newValues{
            rewriter.create<mlir::ConstantOp>(loc, a)};
        rewriter.replaceOp(op, newValues);
        return success();
      } else {
        auto a = op->getAttrOfType<BoolAttr>("value");
        SmallVector<Value, 8> newValues{constInt(a.getValue(), it.getWidth())};
        rewriter.replaceOp(op, newValues);
        return success();
      }
    }
    // FIXME: support float types
    // if(t.isa<FloatType>()) {
    //   APFloat f = *(a.float_value_begin());
    //   rewriter.replaceOp(op, constFloat(f));
    //   return success();
    // }
    return failure();
  }
};

/// Lower Add
class AddOpConversion : public ConversionPattern {
public:
  explicit AddOpConversion(MLIRContext *context)
      : ConversionPattern(mlir::NPCOMP::aten::AddOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter, "add");
  }
};

/// Lower Addmm
class AddmmOpConversion : public ConversionPattern {
public:
  explicit AddmmOpConversion(MLIRContext *context)
      : ConversionPattern(mlir::NPCOMP::aten::AddmmOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter, "addmm");
  }
};

/// Lower AsStrided
class AsStridedOpConversion : public ConversionPattern {
public:
  explicit AsStridedOpConversion(MLIRContext *context)
      : ConversionPattern(mlir::NPCOMP::aten::AsStridedOp::getOperationName(),
                          1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    Value xVal = memRefTypeCast(rewriter, operands[0]);

    // construct the shape argument
    std::vector<Value> shape;
    std::vector<int64_t> result_shape;
    auto co0 =
        cast<mlir::NPCOMP::aten::ConstantOp>(operands[1].getDefiningOp());
    DenseElementsAttr a0 =
        co0.template getAttrOfType<DenseElementsAttr>("value");
    for (auto i : a0.getAttributeValues())
      shape.push_back(rewriter.create<mlir::ConstantOp>(co0.getLoc(), i));

    // pad out the shape with -1 to make it 4d
    while (shape.size() < 4)
      shape.push_back(constInt(-1, 32));

    // construct the stride argument
    std::vector<Value> stride;
    auto co1 =
        cast<mlir::NPCOMP::aten::ConstantOp>(operands[2].getDefiningOp());
    DenseElementsAttr a1 =
        co1.template getAttrOfType<DenseElementsAttr>("value");
    for (auto i : a1.getAttributeValues())
      stride.push_back(rewriter.create<mlir::ConstantOp>(co1.getLoc(), i));

    // pad out the stride with -1 to make it 4d
    while (stride.size() < 4)
      stride.push_back(constInt(-1, 32));

    APInt offset(32, 0);
    if (operands.size() > 3) {
      auto co2 =
          cast<mlir::NPCOMP::aten::ConstantOp>(operands[3].getDefiningOp());
      auto ia2 = co2.getAttrOfType<IntegerAttr>("value");
      offset = ia2.getValue();
    }

    SmallVector<Value, 8> callops{
        xVal,      shape[0],
        shape[1],  shape[2],
        shape[3],  stride[0],
        stride[1], stride[2],
        stride[3], constInt(offset.getSExtValue(), 32)};

    return rewriteWithVoidFunctionCallExplicit(op, callops, operands, rewriter,
                                               "as_strided");
  }
};

/// Lower batchnorm
class BatchNormOpConversion : public ConversionPattern {
public:
  explicit BatchNormOpConversion(MLIRContext *context)
      : ConversionPattern(mlir::NPCOMP::aten::BatchNormOp::getOperationName(),
                          1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter, "batch_norm");
  }
};

/// Lower conv2d
class ConvolutionOpConversion : public ConversionPattern {
public:
  explicit ConvolutionOpConversion(MLIRContext *context)
      : ConversionPattern(mlir::NPCOMP::aten::ConvolutionOp::getOperationName(),
                          1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter, "conv2d");
  }
};

/// Lower conv2d backward
class ConvolutionBackwardOpConversion : public ConversionPattern {
public:
  explicit ConvolutionBackwardOpConversion(MLIRContext *context)
      : ConversionPattern(
            mlir::NPCOMP::aten::ConvolutionBackwardOp::getOperationName(), 1,
            context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter, "conv2d_backward");
  }
};

/// Lower Div
class DivOpConversion : public ConversionPattern {
public:
  explicit DivOpConversion(MLIRContext *context)
      : ConversionPattern(mlir::NPCOMP::aten::DivOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter, "div");
  }
};
/// Lower LogSoftmax
class LogSoftmaxOpConversion : public ConversionPattern {
public:
  explicit LogSoftmaxOpConversion(MLIRContext *context)
      : ConversionPattern(mlir::NPCOMP::aten::LogSoftmaxOp::getOperationName(),
                          1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter, "log_softmax");
  }
};

/// Lower LogSoftmaxBackwardData
class LogSoftmaxBackwardDataOpConversion : public ConversionPattern {
public:
  explicit LogSoftmaxBackwardDataOpConversion(MLIRContext *context)
      : ConversionPattern(
            mlir::NPCOMP::aten::LogSoftmaxBackwardDataOp::getOperationName(), 1,
            context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter,
                                   "log_softmax_backward_data");
  }
};

/// Lower maxpool2d
class MaxPoolOpConversion : public ConversionPattern {
public:
  explicit MaxPoolOpConversion(MLIRContext *context)
      : ConversionPattern(mlir::NPCOMP::aten::MaxPool2dOp::getOperationName(),
                          1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter, "max_pool2d");
  }
};

/// Lower maxpool2d
class MaxPool2dWithIndicesOpConversion : public ConversionPattern {
public:
  explicit MaxPool2dWithIndicesOpConversion(MLIRContext *context)
      : ConversionPattern(
            mlir::NPCOMP::aten::MaxPool2dWithIndicesOp::getOperationName(), 1,
            context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter,
                                   "max_pool2d_with_indices");
  }
};

/// Lower max_pool2d_with_indices_backward
class MaxPool2dWithIndicesBackwardOpConversion : public ConversionPattern {
public:
  explicit MaxPool2dWithIndicesBackwardOpConversion(MLIRContext *context)
      : ConversionPattern(mlir::NPCOMP::aten::MaxPool2dWithIndicesBackwardOp::
                              getOperationName(),
                          1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter,
                                   "max_pool2d_with_indices_backward");
  }
};

/// Lower MM
class MMOpConversion : public ConversionPattern {
public:
  explicit MMOpConversion(MLIRContext *context)
      : ConversionPattern(mlir::NPCOMP::aten::MmOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter, "mm");
  }
};

/// Lower Mul
class MulOpConversion : public ConversionPattern {
public:
  explicit MulOpConversion(MLIRContext *context)
      : ConversionPattern(mlir::NPCOMP::aten::MulOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter, "mul");
  }
};

/// Lower batchnorm
class NativeBatchNormOpConversion : public ConversionPattern {
public:
  explicit NativeBatchNormOpConversion(MLIRContext *context)
      : ConversionPattern(
            mlir::NPCOMP::aten::NativeBatchNormOp::getOperationName(), 1,
            context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter, "native_batch_norm");
  }
};

/// lower NLL Loss backward
class NllLoss2dBackwardOpConversion : public ConversionPattern {
public:
  explicit NllLoss2dBackwardOpConversion(MLIRContext *context)
      : ConversionPattern(
            mlir::NPCOMP::aten::NllLoss2dBackwardOp::getOperationName(), 1,
            context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter,
                                   "nll_loss2d_backward");
  }
};

/// lower NLL Loss forward
class NllLoss2dForwardOpConversion : public ConversionPattern {
public:
  explicit NllLoss2dForwardOpConversion(MLIRContext *context)
      : ConversionPattern(
            mlir::NPCOMP::aten::NllLoss2dForwardOp::getOperationName(), 1,
            context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter,
                                   "nll_loss2d_forward");
  }
};

/// lower NLL Loss backward
class NllLossBackwardOpConversion : public ConversionPattern {
public:
  explicit NllLossBackwardOpConversion(MLIRContext *context)
      : ConversionPattern(
            mlir::NPCOMP::aten::NllLossBackwardOp::getOperationName(), 1,
            context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter, "nll_loss_backward");
  }
};

/// lower NLL Loss forward
class NllLossForwardOpConversion : public ConversionPattern {
public:
  explicit NllLossForwardOpConversion(MLIRContext *context)
      : ConversionPattern(
            mlir::NPCOMP::aten::NllLossForwardOp::getOperationName(), 1,
            context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter, "nll_loss_forward");
  }
};

/// Lower ReLU
class ReLUOpConversion : public ConversionPattern {
public:
  explicit ReLUOpConversion(MLIRContext *context)
      : ConversionPattern(mlir::NPCOMP::aten::ReluOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter, "relu");
  }
};

/// Lower ThresholdBackward
class ThresholdBackwardOpConversion : public ConversionPattern {
public:
  explicit ThresholdBackwardOpConversion(MLIRContext *context)
      : ConversionPattern(
            mlir::NPCOMP::aten::ThresholdBackwardOp::getOperationName(), 1,
            context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter,
                                   "threshold_backward");
  }
};

/// Lower transpose
class TransposeOpConversion : public ConversionPattern {
public:
  explicit TransposeOpConversion(MLIRContext *context)
      : ConversionPattern(mlir::NPCOMP::aten::TOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteWithFunctionCall(op, operands, rewriter, "t");
  }
};

/// Lower view
class ViewOpConversion : public ConversionPattern {
public:
  explicit ViewOpConversion(MLIRContext *context)
      : ConversionPattern(mlir::NPCOMP::aten::ViewOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    edsc::ScopedContext scope(rewriter, loc);

    Value xVal = memRefTypeCast(rewriter, operands[0]);

    // construct the shape argument
    SmallVector<Value, 8> shape;
    auto co =
        dyn_cast<mlir::NPCOMP::aten::ConstantOp>(operands[1].getDefiningOp());
    DenseElementsAttr a = co.template getAttrOfType<DenseElementsAttr>("value");
    for (auto i : a.getAttributeValues())
      shape.push_back(rewriter.create<mlir::ConstantOp>(co.getLoc(), i));

    // pad out the shape with -1 to make it 4d
    while (shape.size() < 4)
      shape.push_back(constInt(-1, 32));

    SmallVector<Value, 8> callops{xVal, shape[0], shape[1], shape[2], shape[3]};

    return rewriteWithVoidFunctionCallExplicit(op, callops, operands, rewriter,
                                               "view");
  }
};

/// Convert an ATen type, this gets called for block and region arguments, and
/// attributes.
MemRefType convertTensorType(TensorType tensor) {
  return mlir::MemRefType::get(tensor.getShape(), tensor.getElementType(), {},
                               0);
}

/// Lower ATen to Standard dialect.  Currently most of the lowerings are done
/// through function calls, which are expected to be implemented through an
/// external library and linked into the resulting code.  In the future, the
/// expectation is that the preferred lowering path would go through TCP.
/// FIXME: Audit this for completeness
struct ATenLoweringPass
    : public PassWrapper<ATenLoweringPass, OperationPass<ModuleOp>> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, StandardOpsDialect>();
  }

  void runOnOperation() override {
    LLVMTypeConverter typeConverter(getOperation().getContext());
    typeConverter.addConversion([&](Type type) {
      if (auto tensor = type.dyn_cast<TensorType>())
        return convertTensorType(tensor).cast<Type>();
      return type;
    });

    OwningRewritePatternList acapPatterns;
    auto module = getOperation();
    auto context = module.getContext();

    // c++ patterns
    acapPatterns.insert<
        ConstantOpConversion, AddOpConversion, ConvolutionOpConversion,
        ReLUOpConversion, TransposeOpConversion, BatchNormOpConversion,
        NativeBatchNormOpConversion, MaxPoolOpConversion,
        MaxPool2dWithIndicesOpConversion, AddmmOpConversion, ViewOpConversion,
        MulOpConversion, MMOpConversion, AsStridedOpConversion,
        LogSoftmaxOpConversion, ThresholdBackwardOpConversion,
        MaxPool2dWithIndicesBackwardOpConversion,
        ConvolutionBackwardOpConversion, NllLossForwardOpConversion,
        NllLossBackwardOpConversion, NllLoss2dForwardOpConversion,
        NllLoss2dBackwardOpConversion, LogSoftmaxOpConversion,
        LogSoftmaxBackwardDataOpConversion, DivOpConversion>(context);

    mlir::populateFuncOpTypeConversionPattern(acapPatterns, context,
                                              typeConverter);

    // tablegen patterns
    populateATenToStdPatterns(context, acapPatterns);

    // Perform acap specific lowering.
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect, StandardOpsDialect,
                           scf::SCFDialect>();
    target.addLegalOp<AffineForOp, AffineApplyOp, AffineYieldOp>();
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType());
    });

    if (failed(applyPartialConversion(module, target, acapPatterns))) {
      emitError(UnknownLoc::get(context), "error lowering ATen\n");
      signalPassFailure();
    }

    // remove dead constant ops
    for (auto function : getOperation().getOps<FuncOp>()) {
      function.walk([&](Operation *op) {
        auto constOp = dyn_cast<mlir::NPCOMP::aten::ConstantOp>(op);
        if (!constOp)
          return;
        if (op->use_empty())
          op->erase();
      });
    }
  }
};

} // namespace

namespace mlir {
namespace NPCOMP {
namespace aten {

std::unique_ptr<mlir::Pass> createATenLoweringPass() {
  return std::make_unique<ATenLoweringPass>();
}

} // namespace aten
} // namespace NPCOMP
} // namespace mlir

void mlir::NPCOMP::aten::registerATenLoweringPass() {
  PassRegistration<ATenLoweringPass>("aten-to-std",
                                     "ATen dialect lowering to function calls");
}
