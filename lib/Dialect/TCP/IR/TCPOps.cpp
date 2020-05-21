//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/TCP/IR/TCPOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::tcp;

//===----------------------------------------------------------------------===//
// ShapeObserveErrorOp
//===----------------------------------------------------------------------===//

LogicalResult ShapeObserveErrorOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(NoneType::get(context));
  return success();
}

//===----------------------------------------------------------------------===//
// GetExtentOp
//===----------------------------------------------------------------------===//

LogicalResult GetExtentOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(IndexType::get(context));
  return success();
}

//===----------------------------------------------------------------------===//
// ShapeFromExtentsOp
//===----------------------------------------------------------------------===//

LogicalResult ShapeFromExtentsOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(shape::ShapeType::get(context));
  return success();
}

namespace mlir {
namespace NPCOMP {
namespace tcp {
#define GET_OP_CLASSES
#include "npcomp/Dialect/TCP/IR/TCPOps.cpp.inc"
} // namespace tcp
} // namespace NPCOMP
} // namespace mlir
