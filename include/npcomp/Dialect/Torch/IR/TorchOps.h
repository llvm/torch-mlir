//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_TORCH_IR_TORCHOPS_H
#define NPCOMP_DIALECT_TORCH_IR_TORCHOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "npcomp/Dialect/Torch/IR/TorchTraits.h"
#include "npcomp/Dialect/Torch/IR/TorchTypes.h"
#include "npcomp/Interfaces/Traits.h"

#define GET_OP_CLASSES
#include "npcomp/Dialect/Torch/IR/TorchOps.h.inc"

template <> struct llvm::DenseMapInfo<::mlir::NPCOMP::Torch::SlotOp> {
  using SlotOp = ::mlir::NPCOMP::Torch::SlotOp;
  static SlotOp getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return SlotOp::getFromOpaquePointer(pointer);
  }
  static SlotOp getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return SlotOp::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(SlotOp val) {
    return hash_value(val.getAsOpaquePointer());
  }
  static bool isEqual(SlotOp lhs, SlotOp rhs) { return lhs == rhs; }
};

template <> struct llvm::DenseMapInfo<::mlir::NPCOMP::Torch::NnModuleOp> {
  using NnModuleOp = ::mlir::NPCOMP::Torch::NnModuleOp;
  static NnModuleOp getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return NnModuleOp::getFromOpaquePointer(pointer);
  }
  static NnModuleOp getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return NnModuleOp::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(NnModuleOp val) {
    return hash_value(val.getAsOpaquePointer());
  }
  static bool isEqual(NnModuleOp lhs, NnModuleOp rhs) { return lhs == rhs; }
};

template <> struct llvm::DenseMapInfo<::mlir::NPCOMP::Torch::ClassTypeOp> {
  using ClassTypeOp = ::mlir::NPCOMP::Torch::ClassTypeOp;
  static ClassTypeOp getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return ClassTypeOp::getFromOpaquePointer(pointer);
  }
  static ClassTypeOp getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return ClassTypeOp::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(ClassTypeOp val) {
    return hash_value(val.getAsOpaquePointer());
  }
  static bool isEqual(ClassTypeOp lhs, ClassTypeOp rhs) { return lhs == rhs; }
};

template <> struct llvm::DenseMapInfo<::mlir::NPCOMP::Torch::GlobalSlotOp> {
  using OpTy = ::mlir::NPCOMP::Torch::GlobalSlotOp;
  static OpTy getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return OpTy::getFromOpaquePointer(pointer);
  }
  static OpTy getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return OpTy::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(OpTy val) {
    return hash_value(val.getAsOpaquePointer());
  }
  static bool isEqual(OpTy lhs, OpTy rhs) { return lhs == rhs; }
};

#endif // NPCOMP_DIALECT_TORCH_IR_TORCHOPS_H
