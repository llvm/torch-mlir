//===- func_builder.h -------------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_FRONTENDS_PYTORCH_CSRC_BUILDER_FUNC_BUILDER_H
#define NPCOMP_FRONTENDS_PYTORCH_CSRC_BUILDER_FUNC_BUILDER_H

#include "mlir_utils.h"
#include "torch_to_mlir_utils.h"

#include "mlir-c/IR.h"

#include <ATen/Tensor.h>
#include <ATen/core/function_schema.h>

namespace torch_mlir {

/// Wraps an MlirOperationState, deallocating it on destruction unless if
/// it has been consumed.
class OperationStateHolder {
public:
  OperationStateHolder(const char *name, MlirLocation loc)
      : state(mlirOperationStateGet(toMlirStringRef(name), loc)) {}
  OperationStateHolder(const OperationStateHolder &) = delete;
  OperationStateHolder(OperationStateHolder &&other) = delete;
  ~OperationStateHolder() {
    if (owned) {
      // Destroying is done by creating and then destroying the operation.
      mlirOperationDestroy(createOperation());
    }
  }

  operator MlirOperationState *() { return &state; }

  MlirOperation createOperation() {
    assert(owned && "cannot createOperation on unowned state");
    owned = false;
    return mlirOperationCreate(&state);
  }

private:
  MlirOperationState state;
  bool owned = true;
};

/// Wraps an MlirBlock under construction, primarily tracking the terminator
/// and supporting manipulation of it. The terminator may be null if it has
/// not yet been constructed.
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

/// Builds a kernel call step by step.
class KernelCallBuilder {
public:
  KernelCallBuilder(MlirContext context, MlirLocation loc,
                    const std::string &kernelName,
                    const c10::FunctionSchema &schema);
  void addOperand(MlirValue operand);
  void addResultType(MlirType resultType);
  MlirOperation create();

protected:
  MlirContext context;
  MlirLocation loc;

private:
  void addSchemaAttrs();
  OperationStateHolder state;
  const c10::FunctionSchema &schema;
};

/// Wraps a 'func' MlirOperation and provides facilities for constructing
/// IR from some stream of Torch operations.
class FuncBuilder {
public:
  /// Callback for inserting a function.
  using Inserter = std::function<void(MlirOperation funcOp)>;

  /// Creates a new func op with the given characteristics. The created
  /// operation is not attached. The caller must either destroy it or add it
  /// to a parent.
  static std::unique_ptr<FuncBuilder>
  createFunction(Inserter &inserter, MlirLocation location,
                 const std::string &name, std::vector<MlirType> &inputTypes);

  MlirContext getContext() { return context; }
  MlirOperation getFuncOp() { return funcOp; }

  /// Gets the function's entry block.
  MlirBlock getEntryBlock() { return entryBlock.getBlock(); }
  BlockBuilder &getEntryBlockBuilder() { return entryBlock; }

  /// Rewrites the function's signature to return the given types. It is
  /// assumed that a compatible terminator has been added.
  void rewriteFuncReturnTypes(std::vector<MlirType> &resultTypes);

  /// Maps a live Tensor to an MlirValue.
  void mapTensor(at::Tensor tensor, MlirValue value) {
    tensorValueMap.push_back(std::make_pair(tensor, value));
  }

  /// Looks up a current mapping of tensor to an MlirValue, returning a null
  /// value if not found.
  MlirValue lookupTensor(at::Tensor tensor);

  /// Gets a scalar constant value.
  MlirValue getScalarConstant(MlirLocation loc, at::Scalar s);

  /// Gets a bool constant value.
  MlirValue getBoolConstant(MlirLocation loc, bool v);

  /// Gets a None constant value.
  MlirValue getNoneConstant(MlirLocation loc);

  /// Gets a general constant value representing the given value
  /// attribute.
  MlirValue getGeneralConstant(MlirLocation loc, MlirAttribute value);

  /// Builds a list with the given elements
  MlirValue buildList(MlirLocation loc, std::vector<MlirValue> &elements);

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
  std::vector<std::pair<at::Tensor, MlirValue>> tensorValueMap;
};

} // namespace torch_mlir

#endif // NPCOMP_FRONTENDS_PYTORCH_CSRC_BUILDER_FUNC_BUILDER_H
