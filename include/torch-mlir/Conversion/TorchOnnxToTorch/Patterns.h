//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_TORCHONNX_CONVERSION_UTILS_H
#define TORCHMLIR_CONVERSION_TORCHONNX_CONVERSION_UTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace mlir::torch::onnx_c {

/// Used during ONNX pattern matching to bind common patterns of operands,
/// result types and attributes to local variables in a way that is easy
/// to fail the pattern if constraints are violated. Most methods return
/// a ParseResult, which allows for chaining like:
///
/// if (binder.tensorOperand(foo) || binder.tensorResultType(t))
///   return failure();
struct OpBinder {
  OpBinder(Operation *op) : op(op) {}

  Location getLoc() { return op->getLoc(); }

  int getNumOperands() { return op->getNumOperands(); }

  // Operand matches of different arities.
  ParseResult tensorOperand(Value &value0) {
    if (op->getNumOperands() != 1)
      return failure();
    value0 = op->getOperand(0);
    if (!toValidTensorType(value0.getType()))
      return failure();
    return success();
  }

  ParseResult tensorOperands(Value &value0, Value &value1) {
    if (op->getNumOperands() != 2)
      return failure();
    value0 = op->getOperand(0);
    value1 = op->getOperand(1);
    if (!toValidTensorType(value0.getType()) ||
        !toValidTensorType(value1.getType()))
      return failure();
    return success();
  }

  ParseResult tensorOperands(SmallVector<Value> &valueList,
                             int64_t numOperands) {
    if (op->getNumOperands() != numOperands)
      return failure();
    for (int64_t i = 0; i < numOperands; i++) {
      Value curr = op->getOperand(i);
      if (!toValidTensorType(curr.getType())) {
        return failure();
      }
      valueList.push_back(curr);
    }
    return success();
  }

  ParseResult tensorOperandAtIndex(Value &valueIdx, int64_t idx) {
    if (idx >= op->getNumOperands())
      return failure();
    valueIdx = op->getOperand(idx);
    if (!toValidTensorType(valueIdx.getType()))
      return failure();
    return success();
  }

  ParseResult tensorOperandsList(llvm::SmallVectorImpl<Value> &values) {
    for (uint32_t i = 0; i < op->getNumOperands(); i++) {
      values.push_back(op->getOperand(i));
    }
    return success();
  }

  // Result type matchers of different arities.
  ParseResult tensorResultType(Torch::ValueTensorType &type0) {
    if (op->getNumResults() != 1)
      return failure();
    auto t = toValidTensorType(op->getResult(0).getType());
    if (!t)
      return failure();
    type0 = t;
    return success();
  }

  ParseResult tensorResultTypeAtIndex(Torch::ValueTensorType &typeIdx,
                                      int64_t idx) {
    if (idx >= op->getNumResults())
      return failure();
    auto t = toValidTensorType(op->getResult(idx).getType());
    if (!t)
      return failure();
    typeIdx = t;
    return success();
  }

  // Attribute accessors.
  ParseResult s64BoolAttr(bool &value, StringRef nameSuffix,
                          bool defaultValue = false) {
    SmallString<64> name("torch.onnx.");
    name.append(nameSuffix);
    auto attr = op->getAttr(name);
    if (!attr) {
      value = defaultValue;
      return success();
    }
    if (auto integerAttr = dyn_cast<IntegerAttr>(attr)) {
      IntegerType t = cast<IntegerType>(integerAttr.getType());
      if (!t.isSigned() || t.getWidth() != 64)
        return failure();
      value = static_cast<bool>(integerAttr.getSInt());
      return success();
    }
    return failure();
  }

  ParseResult s64IntegerAttr(int64_t &value, StringRef nameSuffix,
                             int64_t defaultValue = 0) {
    SmallString<64> name("torch.onnx.");
    name.append(nameSuffix);
    auto attr = op->getAttr(name);
    if (!attr) {
      value = defaultValue;
      return success();
    }
    if (auto integerAttr = dyn_cast<IntegerAttr>(attr)) {
      IntegerType t = cast<IntegerType>(integerAttr.getType());
      if (!t.isSigned() || t.getWidth() != 64)
        return failure();
      value = integerAttr.getSInt();
      return success();
    }
    return failure();
  }

  ParseResult f32FloatAttr(float &value, StringRef nameSuffix,
                           float defaultValue = 0.0f) {
    SmallString<64> name("torch.onnx.");
    name.append(nameSuffix);
    auto attr = op->getAttr(name);
    if (!attr) {
      value = defaultValue;
      return success();
    }
    if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
      FloatType t = cast<FloatType>(floatAttr.getType());
      if (t.getWidth() != 32)
        return failure();
      value = floatAttr.getValue().convertToFloat();
      return success();
    }
    return failure();
  }

  ParseResult s64IntegerArrayAttr(llvm::SmallVector<int64_t> &values,
                                  StringRef nameSuffix,
                                  ArrayRef<int64_t> defaults) {
    SmallString<64> name("torch.onnx.");
    name.append(nameSuffix);
    auto attr = op->getAttr(name);
    if (!attr) {
      values.append(defaults.begin(), defaults.end());
      return success();
    }
    if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
      for (auto element : arrayAttr) {
        auto integerAttr = element.dyn_cast<IntegerAttr>();
        if (!integerAttr)
          return failure();
        IntegerType t = cast<IntegerType>(integerAttr.getType());
        if (!t.isSigned() || t.getWidth() != 64)
          return failure();
        values.push_back(integerAttr.getSInt());
      }
      return success();
    }
    return failure();
  }

  ParseResult denseElementsAttr(ElementsAttr elementsattr,
                                StringRef nameSuffix) {
    SmallString<64> name("torch.onnx.");
    name.append(nameSuffix);
    Attribute attr = op->getAttr(name);
    if (!attr || !isa<ElementsAttr>(attr)) {
      return failure();
    }

    elementsattr = cast<ElementsAttr>(attr);
    return success();
  }

  ParseResult customOpNameStringAttr(std::string &value, StringRef nameSuffix,
                                     std::string defaultValue = "") {
    SmallString<64> name("torch.onnx.");
    name.append(nameSuffix);
    auto attr = op->getAttr(name);
    if (!attr) {
      value = defaultValue;
      return success();
    }
    if (auto stringAttr = dyn_cast<StringAttr>(attr)) {
      value = stringAttr.str();
      return success();
    }
    return failure();
  }

  Torch::ValueTensorType toValidTensorType(Type t) {
    auto tt = dyn_cast<Torch::ValueTensorType>(t);
    if (tt && tt.hasSizes())
      return tt;
    return {};
  }

  Operation *op;
};

/// We use a single pattern per ONNX domain to handle all named custom
/// ops.
/// This allows us to avoid the n^2 problem on pattern application by
/// implementing a secondary index based on the name and sinceVersion
/// attributes.
/// It also lets us add some ergonomics for trivial cases.
class OnnxCustomOpConversionPattern
    : public OpConversionPattern<Torch::OperatorOp> {
public:
  using HandlerFn = LogicalResult (*)(OpBinder binder,
                                      ConversionPatternRewriter &rewriter);
  struct HandlerReg {
    HandlerReg(HandlerFn callback, int64_t sinceVersion)
        : callback(callback), sinceVersion(sinceVersion) {}
    HandlerFn callback;
    int64_t sinceVersion;
  };

  OnnxCustomOpConversionPattern(MLIRContext *context, std::string domainPrefix,
                                int64_t domainVersion)
      : OpConversionPattern(context), domainPrefix(std::move(domainPrefix)),
        domainVersion(domainVersion) {
    // Onnx lowerings could produce other Onnx operations during the rewrite.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult
  matchAndRewrite(Torch::OperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

  /// Adds all fully qualified operator names to the given set.
  /// This is typically used for implementing a dynamic legality
  /// check for torch.operator names.
  void populateLegalizedNames(DenseSet<StringAttr> &legalizedNames);

  /// Register a conversion for a specific ONNX operator. For the
  /// default domain, this is the canonical ONNX operator name (i.e.
  /// "Acos").
  /// Multiple conversions can be registered for the same op, most
  /// commonly differing by their `sinceVersion`.
  void onOp(StringRef name, int64_t sinceVersion, HandlerFn callback);

private:
  std::string domainPrefix;
  int64_t domainVersion;
  DenseMap<StringAttr, SmallVector<HandlerReg, 1>> namedHandlers;
};

// Patterns are split into chunks to speed compile time and reduce some
// contention on the same source files.
void populateDefaultDomainAtoF(OnnxCustomOpConversionPattern &patterns);
void populateDefaultDomainGtoP(OnnxCustomOpConversionPattern &patterns);
void populateDefaultDomainQtoZ(OnnxCustomOpConversionPattern &patterns);

} // namespace mlir::torch::onnx_c

#endif // TORCHMLIR_CONVERSION_TORCHONNX_CONVERSION_UTILS_H
