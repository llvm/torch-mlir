//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/RD/IR/RDOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"
#include "npcomp/Dialect/RD/IR/RDDialect.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::rd;

#define GET_OP_CLASSES
#include "npcomp/Dialect/RD/IR/RDOps.cpp.inc"

llvm::Optional<LLVM::LLVMType> RangeOp::buildStateLLVMType() {
    auto int64Ty = LLVM::LLVMType::getInt64Ty(getContext());
    return LLVM::LLVMType::getStructTy(getContext(), {int64Ty, int64Ty, int64Ty});
}

void RangeOp::buildInitState(OpBuilder& builder, Value ptr, InitArgMap args) {
    auto startValue = args[start()];
    auto endValue = args[end()];
    auto stepValue = args[step()];
    auto int64Ty = LLVM::LLVMType::getInt64Ty(getContext());
    
    auto zero = builder.create<LLVM::ConstantOp>(
        getLoc(), int64Ty, builder.getIntegerAttr(builder.getIndexType(), 0));
    auto one = builder.create<LLVM::ConstantOp>(
        getLoc(), int64Ty, builder.getIntegerAttr(builder.getIndexType(), 1));
    auto two = builder.create<LLVM::ConstantOp>(
        getLoc(), int64Ty, builder.getIntegerAttr(builder.getIndexType(), 2));

    auto startPtr = builder.create<LLVM::GEPOp>(getLoc(), int64Ty.getPointerTo(), ptr, ValueRange({zero, zero}));
    builder.create<LLVM::StoreOp>(getLoc(), startValue, startPtr);

    auto endPtr = builder.create<LLVM::GEPOp>(getLoc(), int64Ty.getPointerTo(), ptr, ValueRange({zero, one}));
    builder.create<LLVM::StoreOp>(getLoc(), endValue, endPtr);

    auto stepPtr = builder.create<LLVM::GEPOp>(getLoc(), int64Ty.getPointerTo(), ptr, ValueRange({zero, two}));
    builder.create<LLVM::StoreOp>(getLoc(), stepValue, stepPtr);
}

llvm::Optional<LLVM::LLVMType> FilterOp::buildStateLLVMType() {
    return {};
}
void FilterOp::buildInitState(OpBuilder &builder, Value ptr, InitArgMap args) {}

llvm::Optional<LLVM::LLVMType> InlineMapOp::buildStateLLVMType() {
    return {};
}
void InlineMapOp::buildInitState(OpBuilder &builder, Value ptr, InitArgMap args) {}
