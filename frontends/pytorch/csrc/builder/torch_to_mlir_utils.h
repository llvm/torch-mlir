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

/// Creates a FunctionType suitable for expressing the signature of `block`.
///
/// `mlir::Block` only has a formalized notion of argument types (bb args), but
/// the exact nature of the block's terminator is left opaque (for example, you
/// can have a weird terminator that "returns all but the first operand").
/// `torch::jit::Block` on the other hand has a formalized notion of a
/// `param_node` and `return_node`, which are effectively dummy operations at
/// the start and end of the block, which establish a formal signature for the
/// block and can be generically reasoned about -- that is what we anchor on
/// here.
MlirType getFunctionTypeFromBlock(MlirContext context,
                                  torch::jit::Block *block);

/// Creates an appropriate MlirAttribute that holds the same values as `tensor`.
MlirAttribute converTensorToMlirElementsAttr(at::Tensor tensor,
                                             MlirLocation loc);

MlirAttribute importAttribute(MlirLocation loc, torch::jit::Node *node,
                              c10::Symbol symbol);

MlirLocation getMlirLocationFromNode(MlirContext context,
                                     torch::jit::Node *node);

std::vector<MlirType>
getMlirTypesFromValues(MlirLocation loc,
                       c10::ArrayRef<torch::jit::Value *> values);

} // namespace torch_mlir

#endif // NPCOMP_FRONTENDS_PYTORCH_CSRC_TORCH_TO_MLIR_UTILS_H
