//===- func_builder.h -------------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_FRONTENDS_PYTORCH_CSRC_C10_DISPATCH_FUNC_BUILDER_H
#define NPCOMP_FRONTENDS_PYTORCH_CSRC_C10_DISPATCH_FUNC_BUILDER_H

#include "mlir-c/IR.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <ATen/Tensor.h>

namespace torch_mlir {

/// Wraps an MlirOperationState, deallocating it on destruction unless if
/// it has been consumed.
class OperationStateHolder {
public:
  OperationStateHolder(const char *name, MlirLocation loc)
      : state(mlirOperationStateGet(name, loc)) {}
  OperationStateHolder(const OperationStateHolder &) = delete;
  OperationStateHolder(OperationStateHolder &&other) = delete;
  ~OperationStateHolder() {
    if (owned) {
      // TODO: Upstream a mlirOperationStateDestroy() function.
      mlirOperationDestroy(createOperation());
    }
  }

  MlirOperation createOperation() {
    assert(owned && "cannot createOperation on unowned state");
    owned = false;
    return mlirOperationCreate(&state);
  }

  MlirOperationState state;

private:
  bool owned = true;
};

/// Maps various runtime types to MlirType.
class TypeMapper {
public:
  TypeMapper(MlirContext context) : context(context) {}

  /// Gets a corresponding MlirType for the Torch ScalarType.
  /// Throws std::invalid_argument on failure.
  MlirType mapScalarType(c10::ScalarType scalarType);

  /// Gets a corresponding MlirType for the forward component of a tensor.
  /// Throws std::invalid_argument on failure.
  MlirType forwardTensorToType(at::Tensor tensor);

private:
  MlirContext context;
};

/// Wraps an MlirBlock under construction, primarily tracking the terminator
/// and supporting manipulation of it. The terminator may be null if it has
/// not yet been constructed, although, for entry blocks, we always construct
/// the function with an appropriate return terminator (which can be changed
/// later).
class BlockBuilder {
public:
  BlockBuilder(MlirBlock block, MlirOperation terminator, bool isReturn)
      : block(block), terminator(terminator), isReturn(isReturn) {}

  MlirBlock getBlock() { return block; }
  MlirOperation getTerminator() { return terminator; }
  bool getIsReturnTerminator() { return isReturn; }

  /// Inserts an owned operation before the terminator.
  void insertBeforeTerminator(MlirOperation op) {
    mlirBlockInsertOwnedOperationBefore(block, terminator, op);
  }

private:
  MlirBlock block;
  MlirOperation terminator;
  bool isReturn;
};

/// Wraps a 'func' MlirOperation and provides facilities for constructing
/// IR from some stream of Torch operations.
class FuncBuilder {
public:
  /// Creates a new func op with the given characteristics. The created
  /// operation is not attached. The caller must either destroy it or add it
  /// to a parent.
  static std::unique_ptr<FuncBuilder>
  createFunction(MlirContext context, MlirLocation location,
                 llvm::StringRef name,
                 llvm::SmallVectorImpl<MlirType> &inputTypes);

  MlirContext getContext() { return context; }
  MlirOperation getFuncOp() { return funcOp; }

  /// Gets the function's entry block.
  MlirBlock getEntryBlock() { return entryBlock.getBlock(); }
  BlockBuilder &getEntryBlockBuilder() { return entryBlock; }

  /// Rewrites the function's signature to return the given types. It is
  /// assumed that a compatible terminator has been added.
  void rewriteFuncReturnTypes(llvm::SmallVectorImpl<MlirType> &resultTypes);

  /// Maps a live Tensor to an MlirValue.
  void mapTensor(at::Tensor tensor, MlirValue value) {
    tensorValueMap.push_back(std::make_pair(tensor, value));
  }

  /// Looks up a current mapping of tensor to an MlirValue, returning a null
  /// value if not found.
  MlirValue lookupTensor(at::Tensor tensor);

  /// Gets a scalar constant value.
  MlirValue getScalarConstant(MlirLocation loc, at::Scalar s);

private:
  FuncBuilder(MlirContext context, MlirOperation funcOp,
              BlockBuilder entryBlock)
      : context(context), funcOp(funcOp), entryBlock(std::move(entryBlock)) {
    (void)this->context;
  }

  /// Inserts a constant op into the function, returning its first result.
  /// The function maintains a constant area at the top where these are
  /// inserted.
  MlirValue insertConstantOp(MlirOperation op);

  MlirContext context;

  /// The func op under construction.
  MlirOperation funcOp;

  /// Block builder for the entry block.
  BlockBuilder entryBlock;

  /// Previously inserted constant op or null.
  MlirOperation prevConstantOp = {nullptr};

  /// Maps tensors to MlirValue. Unfortunately, this needs to be a linear scan
  /// because the impl pointer for the Tensor is not accessible. To make this
  /// slightly better, we add to the back and lookup in reverse under the idea
  /// that tensors may be mapped and accessed in proximity.
  /// TODO: Tensors referenced via an IValue support hash code lookup and
  /// identity checks. Switch to this instead of a linear scan.
  llvm::SmallVector<std::pair<at::Tensor, MlirValue>, 16> tensorValueMap;
};

} // namespace torch_mlir

#endif // NPCOMP_FRONTENDS_PYTORCH_CSRC_C10_DISPATCH_MODULE_BUILDER_H
