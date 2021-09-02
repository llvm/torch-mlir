//===- AnnotateABI.cpp -------------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "iree-dialects/Dialect/IREE/IREEOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "npcomp/Dialect/TorchConversion/Transforms/Passes.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::TorchConversion;
namespace json = llvm::json;

static json::Value
convertTypeToIREEABIJSON(Type type,
                         llvm::function_ref<InFlightDiagnostic()> emitError) {
  if (auto tensorType = type.dyn_cast<Torch::BaseTensorType>()) {
    // TODO: Support unranked and unknown dtype when we actually have examples
    // that need it.
    if (tensorType.hasSizes() && tensorType.hasDtype()) {
      json::Array typeRecord{"ndarray"};
      typeRecord.push_back(
          convertTypeToIREEABIJSON(tensorType.getDtype(), emitError));
      typeRecord.push_back(json::Value(tensorType.getSizes().size()));
      for (auto size : tensorType.getSizes()) {
        if (size == Torch::kUnknownSize)
          typeRecord.push_back(json::Value(nullptr));
        else
          typeRecord.push_back(json::Value(size));
      }
      return typeRecord;
    }
  } else if (auto boolType = type.dyn_cast<Torch::BoolType>()) {
    return json::Value("i1");
  } else if (auto intType = type.dyn_cast<Torch::IntType>()) {
    return json::Value("i64");
  } else if (auto floatType = type.dyn_cast<Torch::FloatType>()) {
    return json::Value("f64");
  } else if (auto listType = type.dyn_cast<Torch::ListType>()) {
    return json::Array{
        json::Value("py_uniform_list"),
        convertTypeToIREEABIJSON(listType.getContainedType(), emitError)};
  } else if (auto dictType = type.dyn_cast<Torch::DictType>()) {
    return json::Array{
        json::Value("py_uniform_dict"),
        convertTypeToIREEABIJSON(dictType.getKeyType(), emitError),
        convertTypeToIREEABIJSON(dictType.getValueType(), emitError)};
  } else if (auto tupleType = type.dyn_cast<Torch::TupleType>()) {
    auto typeRecord = json::Array{"pytuple"};
    for (auto containedType : tupleType.getContainedTypes())
      typeRecord.push_back(convertTypeToIREEABIJSON(containedType, emitError));
    return typeRecord;
  } else if (auto strType = type.dyn_cast<Torch::StringType>()) {
    return json::Value("pystr");
  } else if (auto integerType = type.dyn_cast<mlir::IntegerType>()) {
    // Only used in recursive calls for tensor dtypes.
    return json::Value(("i" + Twine(integerType.getWidth())).str());
  } else if (auto floatType = type.dyn_cast<mlir::FloatType>()) {
    // Only used in recursive calls for tensor dtypes.
    if (floatType.isa<BFloat16Type>())
      return json::Value("bf16");
    return json::Value(("f" + Twine(floatType.getWidth())).str());
  }

  emitError() << "unimplemented: ABI annotation for type " << type;
  return json::Value("error: unimplemented type");
}

namespace {
class AnnotateABIPass : public AnnotateABIBase<AnnotateABIPass> {
  void runOnOperation() override {
    auto module = getOperation();

    bool hadError = false;
    module.walk([&](FuncOp func) {
      if (func.getVisibility() != SymbolTable::Visibility::Public)
        return;
      func.getArgumentTypes();
      json::Array abiArgs;
      json::Array abiResults;
      for (auto type : llvm::enumerate(func.getArgumentTypes())) {
        auto emitError = [&]() {
          hadError = true;
          return func.emitError()
                 << "at function argument " << type.index() << ": ";
        };
        abiArgs.push_back(convertTypeToIREEABIJSON(type.value(), emitError));
      }
      for (auto type : llvm::enumerate(func.getCallableResults())) {
        auto emitError = [&]() {
          hadError = true;
          return func.emitError()
                 << "at function result " << type.index() << ": ";
        };
        abiResults.push_back(convertTypeToIREEABIJSON(type.value(), emitError));
      }

      if (hadError)
        return;

      json::Object abiDict;
      abiDict["v"] = json::Value(1);
      abiDict["a"] = json::Value(std::move(abiArgs));
      abiDict["r"] = json::Value(std::move(abiResults));

      std::string buf;
      llvm::raw_string_ostream os(buf);
      os << json::Value(std::move(abiDict));
      func->setAttr("iree.abi", Builder(func).getStringAttr(os.str()));
    });

    if (hadError)
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::NPCOMP::TorchConversion::createAnnotateABIPass() {
  return std::make_unique<AnnotateABIPass>();
}
