//===- IrHelpers.cpp - Helpers for bridging analysis and IR types ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Typing/Support/CPAIrHelpers.h"

#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "llvm/ADT/Optional.h"

using namespace mlir;
using namespace mlir::NPCOMP::Basicpy;
using namespace mlir::NPCOMP::Typing::CPA;

namespace CPA = mlir::NPCOMP::Typing::CPA;

ObjectValueType::IrTypeConstructor static createTensorLikeIrTypeConstructor(
    TensorType tt) {
  return [tt](ObjectValueType *ovt, llvm::ArrayRef<mlir::Type> fieldTypes,
              MLIRContext *mlirContext,
              llvm::Optional<Location> loc) -> mlir::Type {
    if (auto ranked = tt.dyn_cast<RankedTensorType>()) {
      return RankedTensorType::get(tt.getShape(), fieldTypes.front());
    } else {
      // Unranked.
      return UnrankedTensorType::get(fieldTypes.front());
    }
  };
}

ObjectValueType *CPA::newArrayType(Context &context,
                                   ObjectValueType::IrTypeConstructor irCtor,
                                   Identifier *typeIdentifier,
                                   llvm::Optional<TypeNode *> elementType) {
  TypeNode *concreteElementType;
  if (elementType) {
    concreteElementType = *elementType;
  } else {
    concreteElementType = context.newTypeVar();
  }
  auto arrayElementIdent = context.getIdentifier("e");
  return context.newObjectValueType(irCtor, typeIdentifier, {arrayElementIdent},
                                    {concreteElementType});
}

TypeNode *CPA::getArrayElementType(ObjectValueType *arrayType) {
  assert(arrayType->getFieldCount() == 1 &&
         "expected to be an arity 1 array type");
  return arrayType->getFieldTypes().front();
}

ObjectValueType *CPA::createTensorLikeArrayType(Context &context,
                                                TensorType tensorType) {
  auto elTy = tensorType.getElementType();
  llvm::Optional<TypeNode *> dtype;
  if (elTy != UnknownType::get(tensorType.getContext())) {
    dtype = context.mapIrType(elTy);
  }
  return newArrayType(context, createTensorLikeIrTypeConstructor(tensorType),
                      context.getIdentifier("!Tensor"), dtype);
}

static TypeNode *defaultTypeMapHook(Context &context, mlir::Type irType) {
  // Handle core types that we can't define an interface on.
  if (auto tensorType = irType.dyn_cast<TensorType>()) {
    return createTensorLikeArrayType(context, tensorType);
  }

  return nullptr;
}

Context::IrTypeMapHook CPA::createDefaultTypeMapHook() {
  return defaultTypeMapHook;
}
