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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "npcomp/Dialect/Torch/IR/TorchTraits.h"
#include "npcomp/Dialect/Torch/IR/TorchTypes.h"
#include "npcomp/Interfaces/Traits.h"

#define GET_OP_CLASSES
#include "npcomp/Dialect/Torch/IR/TorchOps.h.inc"

namespace mlir {
namespace NPCOMP {
namespace Torch {

namespace detail {
/// Matches the integer stored in a `torch.constant.int`.
struct torch_constant_int_op_binder {
  int64_t *bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  torch_constant_int_op_binder(int64_t *bv) : bind_value(bv) {}

  bool match(Operation *op) {
    if (auto constantInt = dyn_cast<Torch::ConstantIntOp>(op)) {
      *bind_value = constantInt.value().getSExtValue();
      return true;
    }
    return false;
  }
};
} // namespace detail

/// Matches the integer stored in a `torch.constant.int`.
inline detail::torch_constant_int_op_binder
m_TorchConstantInt(int64_t *bind_value) {
  return detail::torch_constant_int_op_binder(bind_value);
}

namespace detail {
/// Matches the constant integers stored in a `torch.ListConstruct`.
struct torch_list_construct_op_binder {
  SmallVectorImpl<int64_t> &bind_values;

  /// Creates a matcher instance that binds the value to bvs if match succeeds.
  torch_list_construct_op_binder(SmallVectorImpl<int64_t> &bvs)
      : bind_values(bvs) {}

  bool match(Operation *op) {
    if (auto constantNums = dyn_cast<Torch::PrimListConstructOp>(op)) {
      for (Value value : constantNums.elements()) {
        int64_t num;
        if (matchPattern(value, m_TorchConstantInt(&num)))
          bind_values.push_back(num);
        else
          return false;
      }
    }
    return true;
  }
};
} // namespace detail

/// Matches the constant integers stored in a `torch.prim.ListConstruct`.
inline detail::torch_list_construct_op_binder
m_TorchConstantIntList(SmallVectorImpl<int64_t> &bind_values) {
  return detail::torch_list_construct_op_binder(bind_values);
}

/// Create code to copy `tensor` to type `newType`.
///
/// This involves two independent steps, which we keep orthogonal in our
/// IR representation.
/// 1. Adding/removing static information about sizes/dtype.
/// 2. Performing the copy, which allows us to add/remove value semantics.
Value copyTensorToType(OpBuilder &builder, Location loc, BaseTensorType newType,
                       Value tensor);
} // namespace Torch
} // namespace NPCOMP
} // namespace mlir

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
