//===- torch_to_mlir_utils.h ------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_FRONTENDS_PYTORCH_CSRC_TORCH_TO_MLIR_UTILS_H
#define NPCOMP_FRONTENDS_PYTORCH_CSRC_TORCH_TO_MLIR_UTILS_H

#include <memory>

#include "../pybind.h"

#include "mlir-c/IR.h"

#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch_mlir {

/// Maps various runtime types to MlirType.
class TypeMapper {
public:
  TypeMapper(MlirContext context) : context(context) {}

  /// Gets a corresponding MlirType for the Torch ScalarType.
  /// Throws std::invalid_argument on failure.
  MlirType mapFromTorchScalarType(c10::ScalarType scalarType);

  /// Gets a corresponding MlirType for the forward component of a tensor.
  /// Throws std::invalid_argument on failure.
  MlirType forwardTensorToType(at::Tensor tensor);

  /// Gets a corresponding MlirType for the Torch ScalarType.
  /// Torch ScalarType is used to represent the possible element types of Torch
  /// tensors, which is different from the set of types used to represent
  /// Python numeric scalar values (which are always either f64 or i64).
  /// Returns a null type on failure and emits a diagnostic.
  MlirType mapFromTorchScalarType(MlirLocation loc, c10::ScalarType scalarType);

  /// Maps a torch type to a corresponding MlirType. Returns a null type
  /// on failure and emits a diagnostic.
  MlirType mapFromTorchType(MlirLocation loc, const c10::TypePtr &torchType);

private:
  /// Maps from a scalar type and returns null if no match (no other error
  /// reporting).
  MlirType rawMapFromTorchScalarType(c10::ScalarType scalarType);
  MlirContext context;
};

/// Creates a FunctionType suitable for expressing the signature of `schema`.
///
/// This can differ from the type inferred from the block of a
/// torch::jit::Function due to derefinement.
MlirType getFunctionTypeFromSchema(MlirContext context,
                                   const c10::FunctionSchema &schema);

/// Creates an appropriate MlirAttribute that holds the same values as `tensor`.
MlirAttribute convertTensorToMlirElementsAttr(at::Tensor tensor,
                                              MlirLocation loc);

MlirAttribute importAttribute(MlirLocation loc, torch::jit::Node *node,
                              c10::Symbol symbol);

MlirLocation getMlirLocationFromNode(MlirContext context,
                                     torch::jit::Node *node);

std::vector<MlirType>
getMlirTypesFromValues(MlirLocation loc,
                       c10::ArrayRef<torch::jit::Value *> values);

std::vector<MlirValue> derefineValues(c10::ArrayRef<MlirValue> values,
                                      c10::ArrayRef<MlirType> expectedTypes,
                                      MlirLocation loc,
                                      MlirBlock appendToBlock);

/// Create the appropriate MLIR operation for the Torch operator with schema
/// "schema".
///
/// The primary difficulty here is doing the appropriate name munging and
/// checking if the have a registered op.
MlirOperation createOperationFromSchema(MlirBlock appendToBlock,
                                        MlirLocation loc,
                                        const c10::FunctionSchema &schema,
                                        c10::ArrayRef<MlirType> resultTypes,
                                        c10::ArrayRef<MlirValue> operands);

} // namespace torch_mlir

#endif // NPCOMP_FRONTENDS_PYTORCH_CSRC_TORCH_TO_MLIR_UTILS_H
