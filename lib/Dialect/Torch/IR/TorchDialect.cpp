//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Torch/IR/TorchDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/Torch/IR/TorchTypes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

//===----------------------------------------------------------------------===//
// Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct TorchInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool wouldBeCloned,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Tablegen Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "npcomp/Dialect/Torch/IR/TorchTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//

void TorchDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "npcomp/Dialect/Torch/IR/TorchOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "npcomp/Dialect/Torch/IR/TorchTypes.cpp.inc"
      >();
  addInterfaces<TorchInlinerInterface>();
  getContext()->loadDialect<StandardOpsDialect>();
  getContext()->loadDialect<Basicpy::BasicpyDialect>();
}

//===----------------------------------------------------------------------===//
// Type-related Dialect methods.
//===----------------------------------------------------------------------===//

Type TorchDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  Type type;
  if (generatedTypeParser(getContext(), parser, keyword, type).hasValue())
    return type;

  parser.emitError(parser.getNameLoc(), "invalid 'torch' type: `")
      << keyword << "'";
  return Type();
}

void TorchDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (failed(generatedTypePrinter(type, printer)))
    llvm_unreachable("unknown 'torch' type");
}


//===----------------------------------------------------------------------===//
// Dialect-level verifiers.
//===----------------------------------------------------------------------===//

LogicalResult TorchDialect::verifyRegionArgAttribute(Operation *op,
                                                     unsigned regionIndex,
                                                     unsigned argIndex,
                                                     NamedAttribute namedAttr) {
  if (namedAttr.first == "torch.type_bound") {
    auto func = dyn_cast<FuncOp>(op);
    if (!func)
      return op->emitError() << "'torch.type_bound' must be attached to a func";
    TypeAttr attr = namedAttr.second.dyn_cast<TypeAttr>();
    if (!attr)
      return op->emitError() << "'torch.type_bound' must be TypeAttr";
    auto type = attr.getValue().dyn_cast<BaseTensorType>();
    if (!type)
      return op->emitError() << "'torch.type_bound' must be of "
                                "!torch.tensor/!torch.vtensor type";
    if (!func.getType().getInput(argIndex).isa<BaseTensorType>())
      return op->emitError() << "'torch.type_bound' must be attached to an "
                                "argument of !torch.tensor/!torch.vtensor type";
    return success();
  }

  return op->emitError() << "unknown region arg attribute '" << namedAttr.first
                         << "'";
}

//===----------------------------------------------------------------------===//
// Constant materializer.
//===----------------------------------------------------------------------===//

Operation *TorchDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  // Bool (i1 -> !basicpy.BoolType).
  if (type.isa<Basicpy::BoolType>()) {
    auto i1Value = value.dyn_cast<IntegerAttr>();
    if (i1Value && i1Value.getType().getIntOrFloatBitWidth() == 1)
      return builder.create<Basicpy::BoolConstantOp>(loc, type, i1Value);
  }
  // i64 is how we model TorchScript's "scalar integer type" (we could have a
  // proper !torch.int type in theory). None of our canonicalizers should be
  // creating any other integer type (except perhaps i1 after we resolve that
  // situation). All other integer types live inside tensors (that is, they are
  // never the direct result of an operation, and are thus never candidates for
  // constant materialization).
  if (auto integerType = type.dyn_cast<IntegerType>()) {
    if (integerType.getWidth() == 64)
      return builder.create<Torch::ConstantIntOp>(loc,
                                                  value.cast<IntegerAttr>());
  }

  // We currently use the builtin `f64` type to model the Python `float` type.
  // This semantically matches how TorchScript represents it, but is still
  // a little bit ugly.
  // TODO: We should have a !torch.float type to model this.
  if (auto floatType = type.dyn_cast<Float64Type>())
    return builder.create<Torch::ConstantFloatOp>(loc, value.cast<FloatAttr>());

  if (type.isa<Torch::NoneType>())
    return builder.create<ConstantNoneOp>(loc);

  if (auto stringAttr = value.dyn_cast<StringAttr>())
    return builder.create<ConstantStrOp>(loc, stringAttr);

  return nullptr;
}
