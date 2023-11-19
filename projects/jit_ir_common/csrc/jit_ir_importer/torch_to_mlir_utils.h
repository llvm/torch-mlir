//===- torch_to_mlir_utils.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRJITIRIMPORTER_CSRC_TORCH_TO_MLIR_UTILS_H
#define TORCHMLIRJITIRIMPORTER_CSRC_TORCH_TO_MLIR_UTILS_H

#include "import_options.h"

#include <memory>

#include "mlir-c/IR.h"

#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch_mlir {

/// Thrown on failure when details are in MLIR emitted diagnostics.
class mlir_diagnostic_emitted : public std::runtime_error {
public:
  mlir_diagnostic_emitted(const char *what) : std::runtime_error(what) {}
  mlir_diagnostic_emitted() : std::runtime_error("see diagnostics") {}
};

/// Gets a corresponding MlirType for the Torch ScalarType.
/// `c10::`ScalarType` is used to represent tensor dtypes, and is a different
/// type universe from the Python-based types modeled with `c10::Type`.
/// Compared to the Python types, which just have `int` and `float` and
/// `bool` numeric types, ScalarType has detailed bit-width and precision
/// considerations (which matter a lot for tensors, but don't really matter
/// for Python code).
///
/// Returns a null type on failure and emits a diagnostic.
MlirType getMlirTypeForTorchScalarType(MlirLocation loc,
                                       c10::ScalarType scalarType);

/// Maps a torch type to a corresponding MlirType. Returns a null type
/// on failure and emits a diagnostic.
MlirType getMlirTypeFromTorchType(MlirLocation loc,
                                  const c10::TypePtr &torchType,
                                  const ImportOptions &importOptions = {});

/// Creates a FunctionType suitable for expressing the signature of `schema`.
///
/// This can differ from the type inferred from the block of a
/// torch::jit::Function due to derefinement and refinement of tensor types.
MlirType getFunctionTypeFromSchema(MlirContext context,
                                   const c10::FunctionSchema &schema,
                                   const ImportOptions &importOptions = {});

/// Creates an appropriate MlirAttribute that holds the same values as `tensor`.
MlirAttribute convertTensorToMlirElementsAttr(at::Tensor tensor,
                                              MlirLocation loc);

MlirAttribute importAttribute(MlirLocation loc, torch::jit::Node *node,
                              c10::Symbol symbol);

MlirLocation getMlirLocationFromNode(MlirContext context,
                                     torch::jit::Node *node);

std::vector<MlirType>
getMlirTypesFromValues(MlirLocation loc,
                       c10::ArrayRef<torch::jit::Value *> values,
                       const ImportOptions &importOptions = {});

std::vector<MlirValue> adjustStaticInformationForValues(
    MlirBlock appendToBlock, MlirLocation loc, c10::ArrayRef<MlirValue> values,
    c10::ArrayRef<MlirType> desiredTypes, bool userAllowsRefinement);

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

#endif // TORCHMLIRJITIRIMPORTER_CSRC_TORCH_TO_MLIR_UTILS_H
